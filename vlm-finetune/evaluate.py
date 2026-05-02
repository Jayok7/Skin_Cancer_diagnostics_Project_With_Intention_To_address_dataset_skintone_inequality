"""
evaluate.py
Comprehensive evaluation of the fine-tuned model across multiple dimensions:
  1. QC pass rate: does the model's output survive the quality pipeline?
  2. Clinical accuracy: keyword overlap with expected terminology
  3. Spatial fidelity: alignment with Grad-CAM regions
  4. Safety compliance: hedging language presence, no definitive claims
  5. Fluency & length: within target range
  6. DPO win rate: does the DPO model beat the SFT-only model?
"""

import json
import torch
import numpy as np
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from quality_control import (
    check_keywords, check_indeterminate,
    check_spatial_alignment, check_length,
    check_safety_language
)


def load_model(adapter_path):
    """Load model with specified LoRA adapter"""
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(adapter_path)
    return model, processor


def generate_single(model, processor, record):
    """Generate a single response for evaluation"""
    messages = record['messages'][:2]  # System + User only
    
    images = []
    for content in messages[1]['content']:
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
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,        # Near-greedy for evaluation
            do_sample=False,
        )
    
    generated = processor.batch_decode(
        output_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return generated


def evaluate_model(adapter_path, test_jsonl_path, metadata_path, model_name="model"):
    """Full evaluation pipeline for a single model"""
    
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"{'=' * 60}")
    
    model, processor = load_model(adapter_path)
    
    # Load test set
    with open(test_jsonl_path, 'r') as f:
        test_records = [json.loads(line) for line in f]
    
    # Load metadata for QC
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)
    metadata_lookup = {r['image_id']: r for r in all_metadata}
    
    results = {
        'total': 0,
        'qc_passed': 0,
        'keyword_pass': 0,
        'indeterminate_pass': 0,
        'spatial_pass': 0,
        'length_pass': 0,
        'safety_pass': 0,
        'word_counts': [],
        'by_class': defaultdict(lambda: {'total': 0, 'qc_passed': 0}),
        'responses': []
    }
    
    for i, record in enumerate(test_records):
        # Generate response
        response = generate_single(model, processor, record)
        
        # Build temporary record for QC
        # Extract image_id from record path
        image_id = None
        for content in record['messages'][1]['content']:
            if isinstance(content, dict) and content.get('type') == 'image':
                path = content['image'].replace('file://', '')
                image_id = path.split('/')[-1].replace('_overlay.jpg', '').replace('.jpg', '')
                break
        
        if image_id and image_id in metadata_lookup:
            meta = metadata_lookup[image_id]
        else:
            # Fallback: use record data
            meta = {
                'predicted_class': 'unknown',
                'expected_keywords': [],
                'gradcam_spatial_region': 'center'
            }
        
        temp_record = {
            **meta,
            'teacher_reasoning': response
        }
        
        # Run QC checks
        kw_pass, kw_matched = check_keywords(temp_record)
        indet_pass, _ = check_indeterminate(temp_record)
        spatial_pass, spatial_reason = check_spatial_alignment(temp_record)
        len_pass, word_count = check_length(temp_record)
        safety_pass, safety_reason = check_safety_language(temp_record)
        
        all_passed = all([kw_pass, indet_pass, spatial_pass, len_pass, safety_pass])
        
        # Accumulate results
        results['total'] += 1
        results['qc_passed'] += int(all_passed)
        results['keyword_pass'] += int(kw_pass)
        results['indeterminate_pass'] += int(indet_pass)
        results['spatial_pass'] += int(spatial_pass)
        results['length_pass'] += int(len_pass)
        results['safety_pass'] += int(safety_pass)
        results['word_counts'].append(word_count)
        
        pred_class = meta.get('predicted_class', 'unknown')
        results['by_class'][pred_class]['total'] += 1
        results['by_class'][pred_class]['qc_passed'] += int(all_passed)
        
        results['responses'].append({
            'image_id': image_id,
            'response': response,
            'ground_truth': record['messages'][-1]['content'],  # Teacher response
            'qc_passed': all_passed,
            'checks': {
                'keywords': kw_pass,
                'indeterminate': indet_pass,
                'spatial': spatial_pass,
                'length': len_pass,
                'safety': safety_pass
            }
        })
        
        if i % 10 == 0:
            print(f"  Evaluated {i}/{len(test_records)}")
    
    # ── Compute Metrics ──
    n = results['total']
    metrics = {
        'model': model_name,
        'total_examples': n,
        'overall_qc_pass_rate': f"{results['qc_passed'] / n * 100:.1f}%",
        'keyword_pass_rate': f"{results['keyword_pass'] / n * 100:.1f}%",
        'indeterminate_pass_rate': f"{results['indeterminate_pass'] / n * 100:.1f}%",
        'spatial_pass_rate': f"{results['spatial_pass'] / n * 100:.1f}%",
        'length_pass_rate': f"{results['length_pass'] / n * 100:.1f}%",
        'safety_pass_rate': f"{results['safety_pass'] / n * 100:.1f}%",
        'avg_word_count': f"{np.mean(results['word_counts']):.1f}",
        'median_word_count': f"{np.median(results['word_counts']):.1f}",
        'per_class_qc_rate': {
            cls: f"{data['qc_passed'] / data['total'] * 100:.1f}%"
            for cls, data in results['by_class'].items()
        }
    }
    
    # Print report
    print(f"\n{'─' * 40}")
    print(f"  RESULTS: {model_name}")
    print(f"{'─' * 40}")
    for k, v in metrics.items():
        if k == 'per_class_qc_rate':
            print(f"  Per-class QC pass rate:")
            for cls, rate in v.items():
                print(f"    {cls:8s}: {rate}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"{'─' * 40}")
    
    return metrics, results


def compare_sft_vs_dpo(test_jsonl_path, metadata_path):
    """
    Head-to-head comparison: SFT-only vs SFT+DPO
    Measures the DPO win rate across all quality dimensions
    """
    
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD: SFT vs SFT+DPO")
    print("=" * 60)
    
    sft_metrics, sft_results = evaluate_model(
        adapter_path="./checkpoints/sft/final",
        test_jsonl_path=test_jsonl_path,
        metadata_path=metadata_path,
        model_name="SFT-only"
    )
    
    dpo_metrics, dpo_results = evaluate_model(
        adapter_path="./checkpoints/dpo/final",
        test_jsonl_path=test_jsonl_path,
        metadata_path=metadata_path,
        model_name="SFT+DPO"
    )
    
    # Pairwise comparison
    wins = {'dpo': 0, 'sft': 0, 'tie': 0}
    
    for sft_resp, dpo_resp in zip(sft_results['responses'], dpo_results['responses']):
        sft_checks_passed = sum(sft_resp['checks'].values())
        dpo_checks_passed = sum(dpo_resp['checks'].values())
        
        if dpo_checks_passed > sft_checks_passed:
            wins['dpo'] += 1
        elif sft_checks_passed > dpo_checks_passed:
            wins['sft'] += 1
        else:
            wins['tie'] += 1
    
    total = sum(wins.values())
    print(f"\n{'=' * 60}")
    print(f"PAIRWISE WIN RATE")
    print(f"{'=' * 60}")
    print(f"  DPO wins:  {wins['dpo']:4d} ({wins['dpo'] / total * 100:.1f}%)")
    print(f"  SFT wins:  {wins['sft']:4d} ({wins['sft'] / total * 100:.1f}%)")
    print(f"  Ties:      {wins['tie']:4d} ({wins['tie'] / total * 100:.1f}%)")
    print(f"{'=' * 60}")
    
    # Save full comparison
    comparison = {
        'sft_metrics': sft_metrics,
        'dpo_metrics': dpo_metrics,
        'pairwise_wins': wins,
        'dpo_win_rate': f"{wins['dpo'] / total * 100:.1f}%"
    }
    
    with open('training_data/evaluation_report.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


if __name__ == '__main__':
    comparison = compare_sft_vs_dpo(
        test_jsonl_path='training_data/formatted/test.jsonl',
        metadata_path='training_data/clean_dataset.json'
    )