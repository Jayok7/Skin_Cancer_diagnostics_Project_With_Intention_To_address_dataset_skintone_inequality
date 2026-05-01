"""
train_dpo.py
Stage 2: Direct Preference Optimisation

Why DPO after SFT:
  A model that hallucinates confidently is MORE dangerous than one that 
  hedges appropriately. SFT teaches the model WHAT to say; DPO teaches 
  it HOW to say it safely. Specifically, DPO:
  
  1. Penalises definitive unsafe claims ("This is melanoma")
  2. Rewards hedged clinical language ("Features consistent with melanoma")
  3. Penalises spatial hallucination (referencing regions not in Grad-CAM)
  4. Rewards inclusion of confidence qualifiers and clinical recommendations
  5. Penalises non-clinical or informal language

Total preference pairs: 500
  - 350 automated (from QC pipeline scoring)
  - 150 manually curated (domain expert review)
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig

# ──────────────────────────────────────────────
# PyTorch 2.5.1 / TRL 1.x Compatibility Patch
# ──────────────────────────────────────────────
# TRL 1.x expects FSDPModule (introduced in PyTorch 2.6).
# Since the cluster runs PyTorch 2.5.1, we mock it to prevent ImportError.
import torch
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    import torch.distributed.fsdp
    class DummyFSDPModule: pass
    torch.distributed.fsdp.FSDPModule = DummyFSDPModule
# ──────────────────────────────────────────────

from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
import json
import os


# ──────────────────────────────────────────────
# Merge and Format DPO Data
# ──────────────────────────────────────────────

def prepare_dpo_dataset():
    """
    Merge automated + manual pairs into DPO format.
    
    Expected format per example:
    {
        "prompt": [system_msg, user_msg],
        "chosen": "preferred response text",
        "rejected": "dispreferred response text"
    }
    """
    
    # Load automated pairs
    auto_pairs = []
    with open('training_data/dpo_automated_pairs.jsonl', 'r') as f:
        for line in f:
            auto_pairs.append(json.loads(line))
    
    # Load manual pairs (after expert has filled them in)
    manual_pairs = []
    if os.path.exists('training_data/dpo_manual_pairs.jsonl'):
        with open('training_data/dpo_manual_pairs.jsonl', 'r') as f:
            for line in f:
                manual_pairs.append(json.loads(line))
    
    print(f'Automated pairs: {len(auto_pairs)}')
    print(f'Manual pairs:    {len(manual_pairs)}')
    print(f'Total:           {len(auto_pairs) + len(manual_pairs)}')
    
    # Combine and format
    all_pairs = auto_pairs + manual_pairs
    
    formatted = {
        'prompt': [],
        'chosen': [],
        'rejected': []
    }
    
    for pair in all_pairs:
        # Format prompt as chat template string
        prompt_messages = pair['prompt']
        formatted['prompt'].append(json.dumps(prompt_messages))
        formatted['chosen'].append(pair['chosen'])
        formatted['rejected'].append(pair['rejected'])
    
    return Dataset.from_dict(formatted)


# ──────────────────────────────────────────────
# Load SFT Model
# ──────────────────────────────────────────────

print("Loading SFT model for DPO training...")

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

# Load the SFT LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./checkpoints/sft/final",
    is_trainable=True  # Allow further training
)

processor = AutoProcessor.from_pretrained("./checkpoints/sft/final")

# We also need a reference model (frozen copy of SFT model)
ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
ref_model = PeftModel.from_pretrained(ref_model, "./checkpoints/sft/final")
ref_model.eval()

# ──────────────────────────────────────────────
# DPO Configuration
# ──────────────────────────────────────────────

from trl import DPOConfig

dpo_dataset = prepare_dpo_dataset()

training_args = DPOConfig(
    output_dir="./checkpoints/dpo",
    
    # Batch size
    per_device_train_batch_size=1,        # DPO processes pairs — memory intensive
    gradient_accumulation_steps=16,       # Effective batch size = 16
    
    # Learning rate — lower than SFT to avoid catastrophic forgetting
    learning_rate=5e-5,                   # 4x lower than SFT's 2e-4
    lr_scheduler_type="cosine",
    warmup_steps=10,                      # ~10% warmup
    
    # Training duration
    num_train_epochs=2,                   # Fewer epochs — DPO converges faster
    
    # Precision
    bf16=True,
    
    # Memory
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=25,
    
    # Saving
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,
    load_best_model_at_end=True,
    
    # Logging
    logging_steps=5,
    report_to="tensorboard",
    
    remove_unused_columns=False,
    seed=42,
    
    # DPO Specific args
    beta=0.1,                             # KL penalty coefficient
    max_length=2048,
)

# ──────────────────────────────────────────────
# Split DPO dataset (90/10)
# ──────────────────────────────────────────────

dpo_split = dpo_dataset.train_test_split(test_size=0.1, seed=42)

print(f"\nDPO Train: {len(dpo_split['train'])} pairs")
print(f"DPO Eval:  {len(dpo_split['test'])} pairs")

# ──────────────────────────────────────────────
# Launch DPO Training
# ──────────────────────────────────────────────

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dpo_split['train'],
    eval_dataset=dpo_split['test'],
    processing_class=processor,
)

print("\n" + "=" * 60)
print("Starting DPO Alignment Training")
print("=" * 60)
print(f"Preference pairs (train): {len(dpo_split['train'])}")
print(f"Preference pairs (eval):  {len(dpo_split['test'])}")
print(f"Beta (KL penalty):        0.1")
print(f"Learning rate:            {training_args.learning_rate}")
print(f"Effective batch size:     {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print("=" * 60 + "\n")

dpo_trainer.train()

# Save final DPO adapter
dpo_trainer.save_model("./checkpoints/dpo/final")
processor.save_pretrained("./checkpoints/dpo/final")
print("\n[OK] DPO training complete. Adapter saved to ./checkpoints/dpo/final")