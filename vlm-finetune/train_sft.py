"""
train_sft.py
Stage 1: Supervised Fine-Tuning with QLoRA on Qwen3-VL-7B

QLoRA Configuration:
  - Rank (r): 64
    Why: balances expressivity vs parameter count. Rank 64 captures enough
    task-specific adaptation for a single-domain medical task without 
    overfitting on ~1500-2000 examples.
    
  - Alpha (α): 128
    Why: alpha/rank = 2.0 scaling factor. Higher alpha amplifies the LoRA
    update relative to the frozen weights, which is helpful when the 
    base model has no prior dermoscopy training.
    
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Why: targets all linear layers in both attention and MLP blocks.
    For a domain shift this large (general VLM → clinical dermatology),
    adapting only attention is insufficient; MLP layers encode factual 
    knowledge that needs adjustment.
    
  - Quantisation: 4-bit NormalFloat (NF4) with double quantisation
    Why: reduces VRAM from ~28GB (bf16) to ~6-8GB, enabling training on 
    a single consumer GPU (RTX 3090/4090 with 24GB).

Loss Masking:
  The DataCollator masks all tokens in the system and user turns 
  (label = -100). The model is trained ONLY on predicting the assistant's 
  response tokens.
  Why: the user turn contains fixed input (images + CNN context) - training 
  the model to predict this would waste capacity on a trivial copy task and 
  could cause the model to parrot user inputs rather than reason.
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,  # Qwen3-VL uses same arch
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

# ──────────────────────────────────────────────
# Model & Quantisation Configuration
# ──────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # Public model - no auth required

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 - optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Quantises the quantisation constants - saves ~0.4GB
)

print("Loading model with 4-bit quantisation...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # PyTorch native - no extra package needed
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for gradient checkpointing

# ──────────────────────────────────────────────
# LoRA Configuration
# ──────────────────────────────────────────────

lora_config = LoraConfig(
    r=64,                    # Rank - number of low-rank matrices
    lora_alpha=128,          # Scaling factor (effective lr multiplier = alpha/r = 2.0)
    lora_dropout=0.05,       # Light dropout to prevent overfitting on small dataset
    bias="none",             # Don't train biases - minimal impact, saves params
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",            # Query projection  (attention)
        "k_proj",            # Key projection    (attention)
        "v_proj",            # Value projection   (attention)
        "o_proj",            # Output projection  (attention)
        "gate_proj",         # Gate projection    (MLP - SwiGLU)
        "up_proj",           # Up projection      (MLP)
        "down_proj",         # Down projection    (MLP)
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~1.5-2% of total parameters are trainable

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

dataset = load_dataset(
    'json',
    data_files={
        'train': 'training_data/formatted/train.jsonl',
        'validation': 'training_data/formatted/validation.jsonl',
    }
)

print(f"Train: {len(dataset['train'])} examples")
print(f"Validation: {len(dataset['validation'])} examples")


# ──────────────────────────────────────────────
# Collator with Loss Masking
# ──────────────────────────────────────────────

class VLMDataCollator:
    """
    Custom collator that:
    1. Processes multimodal inputs (images + text) via the processor
    2. Masks loss on all non-assistant tokens (system + user turns)
    
    Why loss masking: the model should only learn to generate the 
    assistant's diagnostic reasoning. Training on user text would 
    teach it to copy CNN outputs verbatim (a trivial task that wastes
    model capacity). Training on system prompts would teach it to 
    recite instructions.
    """
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, examples):
        texts = []
        images_list = []
        
        for example in examples:
            messages = example['messages']
            
            # Apply chat template to get the full formatted text
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            
            # Collect images from user turn
            images = []
            for msg in messages:
                if msg['role'] == 'user' and isinstance(msg['content'], list):
                    for content in msg['content']:
                        if content.get('type') == 'image':
                            from PIL import Image
                            img_path = content['image'].replace('file://', '')
                            images.append(Image.open(img_path).convert('RGB'))
            images_list.append(images)
        
        # Process all examples
        batch = self.processor(
            text=texts,
            images=images_list if any(images_list) else None,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        
        # ── Loss Masking ──
        labels = batch['input_ids'].clone()
        
        # Find assistant response boundaries and mask everything else
        for i, text in enumerate(texts):
            tokens = batch['input_ids'][i]
            
            # Encode the assistant's response to find its token span
            assistant_text = examples[i]['messages'][-1]['content']
            
            # Find where assistant content starts in the token sequence
            # The chat template places <|im_start|>assistant\n before the response
            assistant_marker = self.processor.tokenizer.encode(
                '<|im_start|>assistant\n',
                add_special_tokens=False
            )
            
            # Search for the marker in tokens
            marker_len = len(assistant_marker)
            found = False
            for j in range(len(tokens) - marker_len):
                if tokens[j:j + marker_len].tolist() == assistant_marker:
                    # Mask everything BEFORE the assistant response
                    labels[i, :j + marker_len] = -100
                    found = True
                    break
            
            if not found:
                # Fallback: mask first 80% as a rough heuristic
                mask_len = int(len(tokens) * 0.8)
                labels[i, :mask_len] = -100
        
        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        batch['labels'] = labels
        return batch


# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────

training_args = SFTConfig(
    output_dir="./checkpoints/sft",
    
    # Batch size & accumulation
    per_device_train_batch_size=2,       # Small batch for VRAM constraints
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,        # Effective batch size = 2 * 8 = 16
    
    # Learning rate
    learning_rate=2e-4,                   # Standard for QLoRA
    lr_scheduler_type="cosine",           # Cosine decay - smooth convergence
    warmup_steps=20,                      # ~5% of typical run
    
    # Training duration
    num_train_epochs=3,                   # 3 epochs for ~1750 examples
    
    # Precision
    bf16=True,                            # bfloat16 mixed precision
    
    # Memory optimisation
    gradient_checkpointing=True,          # Trade compute for memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=50,
    
    # Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,                   # Keep only best 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Logging
    logging_steps=10,
    report_to="tensorboard",
    
    # Other
    dataloader_num_workers=4,
    remove_unused_columns=False,          # Keep image columns
    seed=42,
    
    # SFT Specific args
    max_seq_length=2048,
    dataset_kwargs={"skip_prepare_dataset": True},
)


# ──────────────────────────────────────────────
# Launch Training
# ──────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=VLMDataCollator(processor),
    processing_class=processor,
)

print("\n" + "=" * 60)
print("Starting SFT Training")
print("=" * 60)
print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")
print(f"Total params:     {model.num_parameters():,}")
print(f"Train examples:   {len(dataset['train'])}")
print(f"Eval examples:    {len(dataset['validation'])}")
print(f"Effective batch:  {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print("=" * 60 + "\n")

trainer.train()

# Save the final LoRA adapter
trainer.save_model("./checkpoints/sft/final")
processor.save_pretrained("./checkpoints/sft/final")
print("\n[OK] SFT training complete. Adapter saved to ./checkpoints/sft/final")