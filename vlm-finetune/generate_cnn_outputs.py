"""
generate_cnn_outputs.py
=======================
Produces classification + Grad-CAM + confidence for every image in the dataset.
Uses the same torchvision EfficientNet-B3 architecture as the diagnostics script.

CSF Usage:
    python generate_cnn_outputs.py \
        --model-path outputs/skin_cancer_diagnostics/best_efficientnet_b3_ham10000.pth \
        --image-dir  datasets/Ham10000/HAM10000_images_part_1 \
        --metadata   datasets/Ham10000/HAM10000_metadata.csv \
        --output-dir training_data
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ── Constants ──
IMAGE_SIZE = 300  # EfficientNet-B3 native resolution
NUM_CLASSES = 7

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses / Bowen\'s Disease',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

CLINICAL_KEYWORDS = {
    'akiec': ['actinic', 'keratosis', 'bowen', 'scaly', 'erythematous', 'precancerous'],
    'bcc':   ['basal cell', 'pearly', 'translucent', 'telangiectasia', 'rolled border'],
    'bkl':   ['seborrheic', 'keratosis', 'benign', 'waxy', 'stuck-on'],
    'df':    ['dermatofibroma', 'fibrous', 'dimple sign', 'firm', 'papule'],
    'mel':   ['melanoma', 'asymmetry', 'irregular border', 'pigment network', 'blue-white'],
    'nv':    ['nevus', 'mole', 'symmetric', 'regular', 'benign', 'pigment network'],
    'vasc':  ['vascular', 'angioma', 'cherry', 'red', 'lacuna', 'blood vessel']
}


# ── Preprocessing (matches diagnostics script) ──
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def build_model(model_path, device, num_classes=NUM_CLASSES):
    """
    Build the same EfficientNet-B3 architecture as
    skin_cancer_diagnostics.py and load trained weights.
    """
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features  # 1536
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


# ── Grad-CAM ──
def generate_gradcam(model, img_tensor, class_idx):
    """Generate Grad-CAM heatmap using PyTorch hooks on torchvision EfficientNet."""
    gradients = []
    activations = []

    # Target layer: last block of features (matches diagnostics GradCAM class)
    target_layer = model.features[-1]

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    score = output[0, class_idx]

    model.zero_grad()
    score.backward()

    fh.remove()
    bh.remove()

    grads = gradients[0][0]
    acts  = activations[0][0]
    weights = grads.mean(dim=(1, 2))

    heatmap = torch.zeros(acts.shape[1:], device=acts.device)
    for i, w in enumerate(weights):
        heatmap += w * acts[i]

    heatmap = torch.relu(heatmap).cpu().numpy()
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap


def get_spatial_description(heatmap, threshold=0.5):
    """Converts Grad-CAM heatmap into a spatial region descriptor"""
    h, w = heatmap.shape
    mask = (heatmap >= threshold).astype(np.float32)

    if mask.sum() == 0:
        return 'diffuse'

    ys, xs = np.where(mask > 0)
    cy, cx = np.mean(ys) / h, np.mean(xs) / w

    vertical = 'upper' if cy < 0.33 else ('lower' if cy > 0.66 else 'center')
    horizontal = 'left' if cx < 0.33 else ('right' if cx > 0.66 else 'center')

    if vertical == 'center' and horizontal == 'center':
        return 'center'
    elif vertical == 'center':
        return horizontal
    elif horizontal == 'center':
        return vertical
    else:
        return f'{vertical}-{horizontal}'


def get_heatmap_intensity(heatmap):
    """Categorise overall activation intensity"""
    mean_val = np.mean(heatmap[heatmap > 0.3]) if np.any(heatmap > 0.3) else 0
    if mean_val > 0.7:
        return 'strong'
    elif mean_val > 0.4:
        return 'moderate'
    else:
        return 'weak'


def process_dataset(model, image_dir, metadata_csv, output_dir, device):
    """Process entire dataset and save CNN outputs"""
    import pandas as pd
    df = pd.read_csv(metadata_csv)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/gradcam', exist_ok=True)

    results = []

    for idx, row in df.iterrows():
        image_id = row['image_id']
        ground_truth = row['dx']

        img_path = None
        # Search multiple directories (HAM10000 has two image dirs)
        search_dirs = [image_dir]
        parent = os.path.dirname(image_dir)
        for d in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            candidate_dir = os.path.join(parent, d)
            if os.path.isdir(candidate_dir):
                search_dirs.append(candidate_dir)

        for sdir in search_dirs:
            for ext in ['.jpg', '.png']:
                candidate = os.path.join(sdir, f'{image_id}{ext}')
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path:
                break

        if img_path is None:
            continue

        image = Image.open(img_path).convert('RGB')
        img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        # ── Inference ──
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            predictions = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(predictions))
        pred_label = CLASS_LABELS[pred_idx]
        confidence = float(predictions[pred_idx])

        # ── Grad-CAM ──
        img_tensor_grad = preprocess(image).unsqueeze(0).to(device)
        img_tensor_grad.requires_grad_(True)
        heatmap = generate_gradcam(model, img_tensor_grad, pred_idx)
        if heatmap is None:
            continue

        np.save(f'{output_dir}/gradcam/{image_id}_gradcam.npy', heatmap)

        heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        original_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.5, heatmap_colored, 0.5, 0)
        gradcam_path = f'{output_dir}/gradcam/{image_id}_overlay.jpg'
        cv2.imwrite(gradcam_path, overlay)

        spatial_region = get_spatial_description(heatmap)
        intensity = get_heatmap_intensity(heatmap)

        all_scores = {
            CLASS_LABELS[i]: round(float(predictions[i]), 4)
            for i in range(len(CLASS_LABELS))
        }

        record = {
            'image_id': image_id,
            'image_path': img_path,
            'gradcam_overlay_path': gradcam_path,
            'gradcam_npy_path': f'{output_dir}/gradcam/{image_id}_gradcam.npy',
            'ground_truth': ground_truth,
            'predicted_class': pred_label,
            'predicted_name': CLASS_FULL_NAMES[pred_label],
            'confidence': round(confidence, 4),
            'all_scores': all_scores,
            'gradcam_spatial_region': spatial_region,
            'gradcam_intensity': intensity,
            'expected_keywords': CLINICAL_KEYWORDS[pred_label],
            'cnn_correct': pred_label == ground_truth
        }
        results.append(record)

        if idx % 100 == 0:
            print(f'Processed {idx}/{len(df)}')

    with open(f'{output_dir}/cnn_outputs.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nProcessed {len(results)} images')
    print(f'Output: {output_dir}/cnn_outputs.json')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate CNN classification + Grad-CAM outputs for VLM teacher pipeline'
    )
    parser.add_argument('--model-path',
                        default='outputs/skin_cancer_diagnostics/best_efficientnet_b3_ham10000.pth',
                        help='Path to trained EfficientNet-B3 model weights')
    parser.add_argument('--image-dir',
                        default='datasets/Ham10000/HAM10000_images_part_1',
                        help='Directory containing images')
    parser.add_argument('--metadata',
                        default='datasets/Ham10000/HAM10000_metadata.csv',
                        help='Path to HAM10000 metadata CSV')
    parser.add_argument('--output-dir',
                        default='training_data',
                        help='Output directory for CNN outputs + Grad-CAM')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'])
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    )

    print(f'\n{"="*60}')
    print(f'  CNN Output Generation (EfficientNet-B3)')
    print(f'{"="*60}')
    print(f'  Model:    {args.model_path}')
    print(f'  Images:   {args.image_dir}')
    print(f'  Metadata: {args.metadata}')
    print(f'  Output:   {args.output_dir}')
    print(f'  Device:   {device}')
    print(f'{"="*60}\n')

    model = build_model(args.model_path, device)
    print('  Model loaded ✓\n')

    results = process_dataset(
        model=model,
        image_dir=args.image_dir,
        metadata_csv=args.metadata,
        output_dir=args.output_dir,
        device=device,
    )