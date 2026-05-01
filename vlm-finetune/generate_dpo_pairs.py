"""
generate_dpo_pairs.py
Creates preference pairs for Direct Preference Optimisation.

Preference pair generation strategy: HYBRID (both automated + manual)

Automated pairs (via constraint checker): ~70% of pairs
  - Run the SFT model on the test set
  - Generate 3 responses per input (temperature sampling)
  - Score each response through the QC pipeline
  - "Chosen" = response that passes all QC checks with highest keyword match count
  - "Rejected" = response that fails any QC check (especially safety or spatial)

Manually curated pairs: ~30% of pairs
  - Domain expert reviews 150 examples
  - Writes/selects preferred response and identifies failure modes
  - Focus on edge cases: low confidence, ambiguous lesions, multi-class confusion

Target: 500 preference pairs total
  - 350 automated
  - 150 manually curated

Alignment criteria:
  1. Safety: prefer hedged language over definitive claims
  2. Accuracy: prefer responses with correct clinical terminology
  3. Spatial fidelity: prefer responses aligned with Grad-CAM
  4. Completeness: prefer responses that include confidence qualifiers
     and clinical recommendations
  5. Conciseness: prefer responses within 80-150 word target range
"""

import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import os


def load_sft_model():
    """Load the SFT-trained model for generation"""
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "./checkpoints/sft/final")
    processor = AutoProcessor.from_pretrained("./checkpoints/sft/final")
    return model, processor


def generate_responses(model, processor, record, n=3, temperature=0.7):
    """Generate n diverse responses for a single input"""
    
    messages = [
        {"role": "system", "content": record['messages'][0]['content']},
        {"role": "user", "content": record['messages'][1]['content']},
    ]
    
    # Load images
    images = []
    for content in record['messages'][1]['content']:
        if isinstance(content, dict) and content.get('type') == 'image':
            img_path = content['image'].replace('file://', '')
            images.append(Image.open(img_path).convert('RGB'))
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors='pt'
    ).to(model.device)
    
    responses = []
    for _ in range(n):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
        
        # Decode only the new tokens
        generated = processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        responses.append(generated)
    
    return responses


def score_response(response_text, record):
    """
    Score a response using the QC pipeline.
    Returns (score: float, passed: bool, details: dict)
    """
    # Import QC functions
    from quality_control import (
        check_keywords, check_indeterminate,
        check_spatial_alignment, check_length,
        check_safety_language
    )
    
    # Create a temporary record with this response
    temp_record = {**record, 'teacher_reasoning': response_text}
    
    score = 0.0
    details = {}
    
    # Keyword check (+0.3)
    kw_pass, kw_matched = check_keywords(temp_record)
    details['keywords'] = {'pass': kw_pass, 'matched': len(kw_matched)}
    if kw_pass:
        score += 0.3
    
    # Indeterminate check (+0.1)
    indet_pass, _ = check_indeterminate(temp_record)
    details['indeterminate'] = {'pass': indet_pass}
    if indet_pass:
        score += 0.1
    
    # Spatial alignment (+0.25)
    spatial_pass, spatial_reason = check_spatial_alignment(temp_record)
    details['spatial'] = {'pass': spatial_pass, 'reason': spatial_reason}
    if spatial_pass:
        score += 0.25
    
    # Length check (+0.1)
    len_pass, word_count = check_length(temp_record)
    details['length'] = {'pass': len_pass, 'word_count': word_count}
    if len_pass:
        score += 0.1
    
    # Safety language (+0.25)
    safety_pass, safety_reason = check_safety_language(temp_record)
    details['safety'] = {'pass': safety_pass, 'reason': safety_reason}
    if safety_pass:
        score += 0.25
    
    passed = all([kw_pass, indet_pass, spatial_pass, len_pass, safety_pass])
    
    return score, passed, details


def generate_automated_pairs(test_jsonl_path, output_path, target_pairs=350):
    """
    Generate preference pairs automatically using the QC pipeline.
    For each input, generate multiple responses and pick best/worst.
    """
    model, processor = load_sft_model()
    
    with open(test_jsonl_path, 'r') as f:
        records = [json.loads(line) for line in f]
    
    # Also load the full record metadata for QC checks
    with open('training_data/clean_dataset.json', 'r') as f:
        metadata = {r['image_id']: r for r in json.load(f)}
    
    pairs = []
    
    for i, record in enumerate(records):
        if len(pairs) >= target_pairs:
            break
        
        # Generate diverse responses
        responses = generate_responses(model, processor, record, n=3)
        # Extract image_id from the record's user message
        image_id = None
        for msg in record['messages']:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'image':
                        # e.g. "path/to/ISIC_0012345.jpg" -> "ISIC_0012345"
                        image_path = content['image']
                        image_id = os.path.basename(image_path).split('.')[0]
                        break
        
        # Merge original metadata into the record for QC
        if image_id and image_id in metadata:
            full_record = {**record, **metadata[image_id]}
        else:
            full_record = record
            
        # Score each response
        scored = []
        for resp in responses:
            score, passed, details = score_response(resp, full_record)
            scored.append({
                'text': resp,
                'score': score,
                'passed': passed,
                'details': details
            })
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        best = scored[0]
        worst = scored[-1]
        
        # Only create a pair if there's meaningful separation
        if best['score'] - worst['score'] >= 0.2:
            pair = {
                'prompt': record['messages'][:2],  # System + User
                'chosen': best['text'],
                'rejected': worst['text'],
                'chosen_score': best['score'],
                'rejected_score': worst['score'],
                'chosen_details': best['details'],
                'rejected_details': worst['details'],
                'source': 'automated'
            }
            pairs.append(pair)
        
        if i % 20 == 0:
            print(f'Generated {len(pairs)} pairs from {i} inputs')
    
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f'\nGenerated {len(pairs)} automated preference pairs')
    return pairs


# ──────────────────────────────────────────────
# Manual Curation Template
# ──────────────────────────────────────────────

def create_curation_template(test_jsonl_path, output_path, n=150):
    """
    Creates a spreadsheet-friendly JSON for domain expert review.
    The expert fills in 'chosen' and 'rejected' fields.
    """
    with open(test_jsonl_path, 'r') as f:
        records = [json.loads(line) for line in f][:n]
    
    template = []
    for record in records:
        template.append({
            'id': len(template),
            'user_input': record['messages'][1],
            'model_response': '',       # Will be filled by generating
            'chosen': '',                # Expert writes/selects preferred
            'rejected': '',              # Expert writes/selects rejected
            'rejection_reason': '',      # Expert explains why rejected is worse
            'alignment_criteria': [
                'safety', 'accuracy', 'spatial_fidelity',
                'completeness', 'conciseness'
            ]
        })
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f'Created curation template with {len(template)} examples')


if __name__ == '__main__':
    # Generate automated pairs
    generate_automated_pairs(
        test_jsonl_path='training_data/formatted/test.jsonl',
        output_path='training_data/dpo_automated_pairs.jsonl',
        target_pairs=350
    )
    
    # Create manual curation template
    create_curation_template(
        test_jsonl_path='training_data/formatted/test.jsonl',
        output_path='training_data/dpo_manual_template.json',
        n=150
    )