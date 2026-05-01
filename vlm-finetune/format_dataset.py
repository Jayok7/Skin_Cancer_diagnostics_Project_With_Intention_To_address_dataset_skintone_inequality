"""
format_dataset.py
Converts clean dataset to JSONL conversational format for Qwen3-VL fine-tuning.

Why JSONL conversational format: matches the chat template expected by 
Qwen3-VL's tokenizer and training pipeline. Each turn is explicitly 
marked as 'user' or 'assistant', enabling loss masking on assistant 
tokens only.
"""

import json
import os
import base64
from sklearn.model_selection import train_test_split


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def format_user_message(record):
    """
    Construct the user turn with both images and CNN context.
    The user turn contains:
      - The original dermoscopic image
      - The Grad-CAM overlay image
      - Structured CNN output text
    """
    context = (
        f"You are analyzing a dermoscopic skin lesion image. "
        f"A CNN classifier (EfficientNetV2B2) has produced the following output:\n\n"
        f"**Predicted Classification:** {record['predicted_name']}\n"
        f"**Confidence Score:** {record['confidence'] * 100:.1f}%\n"
        f"**Grad-CAM Focus Region:** {record['gradcam_spatial_region']} "
        f"(intensity: {record['gradcam_intensity']})\n\n"
        f"**All Class Probabilities:**\n"
    )
    
    for cls, score in sorted(
        record['all_scores'].items(), key=lambda x: x[1], reverse=True
    ):
        context += f"  - {cls}: {score * 100:.1f}%\n"
    
    context += (
        f"\nThe first image is the original dermoscopic photograph. "
        f"The second image is the Grad-CAM heatmap overlay showing which "
        f"regions the CNN focused on for its classification.\n\n"
        f"Provide a concise diagnostic reasoning paragraph interpreting "
        f"these findings for a clinician."
    )
    
    return context


def create_conversation(record):
    """
    Create a single conversation in Qwen3-VL format.
    
    Format follows the Qwen2.5-VL / Qwen3-VL fine-tuning spec:
    messages: [
        {role: "system", content: "..."},
        {role: "user", content: [{type: "image", image: "..."}, {type: "text", text: "..."}]},
        {role: "assistant", content: "..."}
    ]
    """
    
    system_message = (
        "You are a clinical AI assistant trained to interpret dermoscopic images "
        "alongside CNN classification outputs and Grad-CAM heatmaps. You provide "
        "concise, evidence-based diagnostic reasoning. You never make definitive "
        "diagnoses. You always recommend clinical correlation."
    )
    
    user_content = format_user_message(record)
    
    conversation = {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.abspath(record['image_path'])
                    },
                    {
                        "type": "image",
                        "image": os.path.abspath(record['gradcam_overlay_path'])
                    },
                    {
                        "type": "text",
                        "text": user_content
                    }
                ]
            },
            {
                "role": "assistant",
                "content": record['teacher_reasoning']
            }
        ]
    }
    
    return conversation


def format_and_split(clean_dataset_path, output_dir):
    """
    Format dataset and split into train/validation/test.
    Split: 85% train, 10% validation, 5% test
    """
    
    with open(clean_dataset_path, 'r') as f:
        records = json.load(f)
    
    # Stratified split by predicted class
    labels = [r['predicted_class'] for r in records]
    
    # First split: 85% train, 15% temp
    train_records, temp_records, train_labels, temp_labels = train_test_split(
        records, labels, test_size=0.15, stratify=labels, random_state=42
    )
    
    # Second split: 10% val, 5% test (from the 15%)
    val_records, test_records = train_test_split(
        temp_records, test_size=0.333, stratify=temp_labels, random_state=42
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        'train': train_records,
        'validation': val_records,
        'test': test_records
    }
    
    for split_name, split_records in splits.items():
        output_file = os.path.join(output_dir, f'{split_name}.jsonl')
        
        with open(output_file, 'w') as f:
            for record in split_records:
                conversation = create_conversation(record)
                f.write(json.dumps(conversation) + '\n')
        
        print(f'{split_name}: {len(split_records)} examples → {output_file}')
    
    print(f'\nTotal: {len(records)} examples')
    print(f'Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}')


if __name__ == '__main__':
    format_and_split(
        clean_dataset_path='training_data/clean_dataset.json',
        output_dir='training_data/formatted'
    )