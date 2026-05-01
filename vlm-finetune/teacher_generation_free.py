"""
teacher_generation_free.py
FREE / OPEN-SOURCE version of the teacher generation pipeline.
=============================================================
Uses Qwen2.5-VL-7B-Instruct (open-source, Apache 2.0 licence) as the
teacher model instead of GPT-4o.  Runs entirely locally on GPU — no API
keys, no cost.

Trade-offs vs. the GPT-4o version (teacher_generation.py):
  ┌─────────────────────┬──────────────────┬─────────────────────┐
  │                     │ GPT-4o (paid)    │ Qwen2.5-VL (free)   │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Cost                │ ~$25-50          │ $0                  │
  │ Quality             │ Higher           │ Slightly lower      │
  │ Speed (2500 imgs)   │ 2-4h (API)       │ 4-8h (local GPU)    │
  │ Hardware needed     │ CPU only         │ 1× GPU (≥16GB VRAM) │
  │ Internet required   │ Yes              │ No                  │
  │ Reproducibility     │ API may change   │ Fully reproducible  │
  └─────────────────────┴──────────────────┴─────────────────────┘

Why Qwen2.5-VL-7B-Instruct:
  - Same model family as the fine-tune target (Qwen3-VL-7B), so the
    teacher's output style naturally aligns with what the student will
    learn. This is a form of *curriculum alignment*.
  - 7B parameters fits on a single GPU with 4-bit quantisation (~6GB).
  - Multimodal: natively accepts interleaved image + text input.
  - Apache 2.0 licence: no restrictions on academic or commercial use.

Usage:
    # On CSF (GPU node):
    python teacher_generation_free.py

    # Or with custom paths:
    python teacher_generation_free.py \
        --cnn-outputs training_data/cnn_outputs.json \
        --output training_data/teacher_outputs_free.json \
        --max-samples 2500
"""

import os
import json
import argparse
import time
import random
from collections import defaultdict

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image


# ══════════════════════════════════════════════════════════════
# TEACHER SYSTEM PROMPT  (identical to GPT-4o version)
# ══════════════════════════════════════════════════════════════

TEACHER_SYSTEM_PROMPT = """You are a board-certified dermatologist interpreting a dermoscopic image \
alongside its AI-generated Grad-CAM heatmap and classification output.

STRICT RULES:
1. Reference ONLY features visible in the provided images. 
2. If the image resolution is too low to see specific features, do not invent them. 
   State that the features are indeterminate.
3. Your spatial references (e.g., "the upper-left region") must align with where 
   the Grad-CAM heatmap shows activation.
4. Always include a confidence qualifier based on the CNN confidence score:
   - >90%: "The model shows high confidence in this classification"
   - 70-90%: "The model shows moderate confidence; differential diagnoses should be considered"
   - <70%: "The model shows low confidence; clinical correlation is strongly recommended"
5. Never state a definitive diagnosis. Always use hedging language: 
   "consistent with", "suggestive of", "features characteristic of"
6. Always end with a recommendation to correlate clinically or biopsy if warranted.
7. Keep responses between 80-150 words.
8. Structure: (a) describe what Grad-CAM highlights, (b) describe visible dermoscopic features,
   (c) state the classification with confidence qualifier, (d) clinical recommendation."""


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_teacher_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct", use_4bit=True):
    """
    Load the open-source teacher VLM with optional 4-bit quantisation.
    
    4-bit quantisation reduces VRAM from ~16GB (bf16) to ~6GB,
    enabling single-GPU training on consumer hardware.
    """
    print(f"\nLoading teacher model: {model_id}")
    print(f"  Quantisation: {'4-bit NF4' if use_4bit else 'bfloat16 (full)'}")
    
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    
    model.eval()
    print("  ✓ Model loaded")
    return model, processor


# ══════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════

def generate_teacher_reasoning(model, processor, record):
    """
    Generate diagnostic reasoning for a single image using the
    open-source Qwen2.5-VL model.
    
    Input format matches the Qwen2.5-VL chat template:
      - System message with clinical instructions
      - User message with two images + CNN context text
    """
    
    user_prompt = f"""Analyze this dermoscopic image with the following CNN classification output:

**Predicted Diagnosis:** {record['predicted_name']}
**Confidence Score:** {record['confidence'] * 100:.1f}%
**CNN Correct vs Ground Truth:** {'Yes' if record['cnn_correct'] else 'No'}

**Top differential scores:**
{json.dumps(record['all_scores'], indent=2)}

**Grad-CAM Analysis:**
- Primary activation region: {record['gradcam_spatial_region']}
- Activation intensity: {record['gradcam_intensity']}

The first image is the original dermoscopic image.
The second image is the Grad-CAM overlay showing which regions the CNN focused on.

Provide a concise diagnostic reasoning paragraph."""

    # Build messages in Qwen2.5-VL format
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(record['image_path'])}"},
                {"type": "image", "image": f"file://{os.path.abspath(record['gradcam_overlay_path'])}"},
                {"type": "text", "text": user_prompt},
            ]
        }
    ]
    
    # Process input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Load images
    images = [
        Image.open(record['image_path']).convert('RGB'),
        Image.open(record['gradcam_overlay_path']).convert('RGB'),
    ]
    
    inputs = processor(
        text=[text],
        images=images,
        return_tensors='pt',
    ).to(model.device)
    
    # Generate with low temperature for clinical consistency
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    
    # Decode only the new tokens
    generated = processor.batch_decode(
        output_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    
    return generated


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def generate_all(cnn_outputs_path, output_path, max_samples=2500,
                 model_id="Qwen/Qwen2.5-VL-7B-Instruct", use_4bit=True):
    """
    Generate teacher reasoning for the entire dataset using the
    open-source Qwen2.5-VL model.
    
    Identical pipeline to teacher_generation.py (GPT-4o version),
    but runs locally with no API cost.
    """
    
    with open(cnn_outputs_path, 'r') as f:
        records = json.load(f)
    
    # Stratified sampling: ensure balanced class representation
    by_class = defaultdict(list)
    for r in records:
        by_class[r['predicted_class']].append(r)
    
    # Target ~350 per class for 7 classes = ~2450 total
    samples_per_class = max_samples // len(by_class)
    selected = []
    for cls, items in by_class.items():
        random.shuffle(items)
        selected.extend(items[:samples_per_class])
    
    random.shuffle(selected)
    print(f'\nSelected {len(selected)} samples for teacher generation')
    print(f'  Classes: {len(by_class)}')
    print(f'  Samples/class: ~{samples_per_class}')
    
    # Load model
    model, processor = load_teacher_model(model_id, use_4bit)
    
    results = []
    errors = 0
    t0 = time.time()
    
    for i, record in enumerate(selected):
        try:
            reasoning = generate_teacher_reasoning(model, processor, record)
            record['teacher_reasoning'] = reasoning
            record['teacher_model'] = model_id  # track which model generated this
            results.append(record)
            
        except Exception as e:
            print(f'  ⚠ Error on {record["image_id"]}: {e}')
            errors += 1
            continue
        
        # Progress reporting
        if (i + 1) % 25 == 0 or (i + 1) == len(selected):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(selected) - i - 1) / max(rate, 1e-6)
            print(f'  [{i+1}/{len(selected)}] '
                  f'{rate:.1f} img/s | '
                  f'ETA: {eta/60:.0f}min | '
                  f'errors: {errors}')
            
            # Checkpoint save every 25
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - t0
    print(f'\n{"="*60}')
    print(f'Teacher Generation Complete (FREE — {model_id})')
    print(f'{"="*60}')
    print(f'  Generated:  {len(results)} / {len(selected)} records')
    print(f'  Errors:     {errors}')
    print(f'  Time:       {total_time/60:.1f} minutes')
    print(f'  Rate:       {len(results)/max(total_time,1):.1f} img/s')
    print(f'  Output:     {output_path}')
    print(f'{"="*60}')
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FREE teacher generation using open-source Qwen2.5-VL'
    )
    parser.add_argument('--cnn-outputs', default='training_data/cnn_outputs.json',
                        help='Path to CNN outputs JSON')
    parser.add_argument('--output', default='training_data/teacher_outputs_free.json',
                        help='Output path for teacher reasoning')
    parser.add_argument('--max-samples', type=int, default=2500,
                        help='Max samples to process (stratified across classes)')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='HuggingFace model ID for the teacher VLM')
    parser.add_argument('--no-4bit', action='store_true',
                        help='Disable 4-bit quantisation (uses more VRAM)')
    args = parser.parse_args()
    
    generate_all(
        cnn_outputs_path=args.cnn_outputs,
        output_path=args.output,
        max_samples=args.max_samples,
        model_id=args.model,
        use_4bit=not args.no_4bit,
    )
