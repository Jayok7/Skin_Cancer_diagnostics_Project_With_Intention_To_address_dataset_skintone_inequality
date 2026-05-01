import os
os.environ["HF_HOME"] = "D:/vlm-for-project" # <-- ADD THIS LINE HERE

import io
import base64
import numpy as np
import tensorflow as tf
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import cv2

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

app = Flask(__name__, static_folder='.')

# Model configuration
IMAGE_SIZE = 260

# HAM10000 Labels (7 Classes)
CLASS_LABELS_HAM = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES_HAM = {
    'akiec': 'Actinic Keratoses', 'bcc': 'Basal Cell Carcinoma', 
    'bkl': 'Benign Keratosis-like Lesions', 'df': 'Dermatofibroma', 
    'mel': 'Melanoma', 'nv': 'Melanocytic Nevus', 'vasc': 'Vascular Lesion'
}

# ISIC 2019 Labels (8 Classes)
CLASS_LABELS_ISIC = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
CLASS_NAMES_ISIC = {
    'AK': 'Actinic Keratosis', 'BCC': 'Basal Cell Carcinoma', 
    'BKL': 'Benign Keratosis', 'DF': 'Dermatofibroma', 
    'MEL': 'Melanoma', 'NV': 'Melanocytic Nevus', 
    'SCC': 'Squamous Cell Carcinoma', 'VASC': 'Vascular Lesion'
}

# Global model variables
model = None
vlm_model = None
vlm_processor = None

# Change your global variable at the top from `model = None` to:
cnn_models = {}
vlm_model = None
vlm_processor = None

def load_models():
    """Load all available CNN models and the VLM"""
    global cnn_models, vlm_model, vlm_processor
    
    # 1. Define where your models MIGHT be. 
    # Update these paths to match wherever you decide to store them!
    model_paths = {
        "HAM10000 Baseline": "CNNs/best_efficientnet_b3_ham10000.pth",
        "ISIC 2019 (Original)": "CNNs/best_efficientnet_b3_isic2019_orig.pth",
        "ISIC 2019 (Augmented λ=0.0)": "CNNs/best_efficientnet_b3_isic2019_aug_v2_00.pth",
        "ISIC 2019 (Augmented λ=0.3)": "CNNs/best_efficientnet_b3_isic2019_aug_v2_03.pth",
        "ISIC 2019 (Augmented λ=0.7)": "CNNs/best_efficientnet_b3_isic2019_aug_v2_07.pth",
        "ISIC 2019 (Augmented λ=1.0)": "CNNs/best_efficientnet_b3_isic2019_aug_v2_10.pth"
    }
    
    print("Loading PyTorch CNN models...")
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                model = models.efficientnet_b3(weights=None)
                num_ftrs = model.classifier[1].in_features
                # Dynamically set output features (8 for ISIC, 7 for HAM)
                out_features = 8 if "ISIC" in name else 7

                model.classifier = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(num_ftrs),
                    torch.nn.Linear(num_ftrs, 256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(256, out_features) # <-- DYNAMIC NOW!
                )
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                model.eval()
                cnn_models[name] = model
                print(f"  [OK] Loaded: {name}")
            except Exception as e:
                print(f"  [ERROR] Failed to load {name}: {str(e)}")
        else:
            print(f"  [SKIP] Not found: {name}")

    if not cnn_models:
        print("[ERROR] No CNN models were found!")
        return False

    # --- Load VLM ---
    try:
        print("\nLoading Qwen2.5-VL model (this may take a minute)...")
        base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        base_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        dpo_checkpoint_path = os.path.join("checkpoints", "final")
        if os.path.exists(dpo_checkpoint_path):
            vlm_model = PeftModel.from_pretrained(base_vlm, dpo_checkpoint_path)
        else:
            print(f"[WARNING] DPO checkpoint not found. Using base model.")
            vlm_model = base_vlm
            
        vlm_processor = AutoProcessor.from_pretrained(base_model_id)
        print("[OK] VLM loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading VLM: {str(e)}")
        
    return True

@app.route('/predict', methods=['POST'])
def predict():
    if not cnn_models:
        return jsonify({'error': 'No CNN Models loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    # Grab the selected model from the frontend, fallback to first available if missing
    model_choice = request.form.get('model_choice')
    if model_choice in cnn_models:
        active_model = cnn_models[model_choice]
    else:
        model_choice = list(cnn_models.keys())[0]
        active_model = cnn_models[model_choice]
    
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        original_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        img_array = preprocess_image(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Make prediction with ACTIVE MODEL
        with torch.no_grad():
            outputs = active_model(img_array)
            probabilities = F.softmax(outputs, dim=1)[0]
            predictions = probabilities.numpy() 
            
        # Swap label dictionaries based on which model is active
        active_labels = CLASS_LABELS_ISIC if "ISIC" in model_choice else CLASS_LABELS_HAM
        active_names = CLASS_NAMES_ISIC if "ISIC" in model_choice else CLASS_NAMES_HAM
            
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_label = active_labels[predicted_class_idx]
        predicted_name = active_names[predicted_label]
        
        # Generate Grad-CAM with ACTIVE MODEL
        heatmap = generate_gradcam(active_model, img_array, predicted_class_idx)
        if heatmap is not None:
            gradcam_overlay = create_heatmap_overlay(original_array, heatmap)
            gradcam_b64 = image_to_base64(gradcam_overlay)
        else:
            gradcam_b64 = None
        
        original_b64 = image_to_base64(original_array)
        all_confidences = {
            CLASS_NAMES[CLASS_LABELS[i]]: float(predictions[i])
            for i in range(len(CLASS_LABELS))
        }
        
        vlm_report = generate_vlm_report(image, predicted_name, confidence)
        
        return jsonify({
            'success': True,
            'model_used': model_choice, # Send back which model was used
            'predicted_class': predicted_label,
            'predicted_name': predicted_name,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'original_image': original_b64,
            'gradcam_image': gradcam_b64,
            'vlm_report': vlm_report
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'available_models': list(cnn_models.keys()), # Send available models to frontend!
        'vlm_loaded': vlm_model is not None
    })

    # Load VLM
    try:
        print("Loading Qwen2.5-VL model (this may take a minute)...")
        # Base model (Full power for CSF GPUs)
        base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        base_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load Peft / DPO checkpoint
        dpo_checkpoint_path = os.path.join("checkpoints", "final")
        if os.path.exists(dpo_checkpoint_path):
            print(f"Loading LoRA weights from {dpo_checkpoint_path}...")
            vlm_model = PeftModel.from_pretrained(base_vlm, dpo_checkpoint_path)
        else:
            print(f"[WARNING] DPO checkpoint not found at {dpo_checkpoint_path}. Using base model.")
            vlm_model = base_vlm
            
        vlm_processor = AutoProcessor.from_pretrained(base_model_id)
        print("[OK] VLM loaded successfully")
        
    except Exception as e:
        print(f"[ERROR] Error loading VLM: {str(e)}")
        print("Continuing without VLM support...")
        
    return True

def preprocess_image(image, target_size):
    """Preprocess image for PyTorch model inference"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transforms and add batch dimension [1, C, H, W]
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def generate_gradcam(model, img_tensor, class_idx):
    """
    Generate Grad-CAM heatmap for PyTorch EfficientNet
    """
    try:
        # EfficientNet-B3 final convolutional feature layer
        target_layer = model.features[-1]
        
        activations = []
        gradients = []

        # Hooks to grab the math during the forward and backward passes
        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hooks
        handle_fw = target_layer.register_forward_hook(forward_hook)
        handle_bw = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        model.eval()
        model.zero_grad()
        output = model(img_tensor)

        # Backward pass for the specific predicted class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        # Remove hooks immediately so they don't stack up on the next web request
        handle_fw.remove()
        handle_bw.remove()

        act = activations[0]
        grad = gradients[0]

        # Calculate the heatmap
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize heatmap to match the input image size
        cam = F.interpolate(cam, size=(img_tensor.shape[2], img_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if np.max(cam) == 0 or np.isnan(np.max(cam)):
            return None

        # Normalize between 0 and 1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
        
    except Exception as e:
        print(f"[Grad-CAM Error]: {str(e)}")
        return None

def create_heatmap_overlay(original_image, heatmap):
    """
    Create a colored heatmap overlay on the original image
    """
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.power(heatmap_resized, 0.7)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.5, heatmap_colored, 0.5, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(image_array.astype('uint8'))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def generate_vlm_report(image, predicted_name, confidence):
    """Generate diagnostic reasoning using the loaded VLM"""
    if vlm_model is None or vlm_processor is None:
        return "VLM Model is not loaded. Cannot generate diagnostic report."
        
    try:
        # Save image temporarily or pass directly if supported
        # For qwen-vl-utils, we can pass the PIL image directly
        
        prompt = f"The CNN model predicted this skin lesion is {predicted_name} with {confidence*100:.1f}% confidence. Analyze the dermoscopic features visible in this image to explain and validate this diagnosis. Provide a clear, clinical description of the structures you see."
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert dermatologist assistant. Analyze skin lesion images clinically and concisely."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(vlm_model.device)
        
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating VLM report: {str(e)}"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'CNN Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        original_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        img_array = preprocess_image(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_array)
            probabilities = F.softmax(outputs, dim=1)[0]
            predictions = probabilities.numpy() # Convert back to numpy for the rest of your code
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        predicted_label = CLASS_LABELS[predicted_class_idx]
        predicted_name = CLASS_NAMES[predicted_label]
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(model, img_array, predicted_class_idx)
        if heatmap is not None:
            gradcam_overlay = create_heatmap_overlay(original_array, heatmap)
            gradcam_b64 = image_to_base64(gradcam_overlay)
        else:
            gradcam_b64 = None
        
        original_b64 = image_to_base64(original_array)
        
        all_confidences = {
            CLASS_NAMES[CLASS_LABELS[i]]: float(predictions[i])
            for i in range(len(CLASS_LABELS))
        }
        
        # Generate VLM Report
        print("Generating VLM reasoning...")
        vlm_report = generate_vlm_report(image, predicted_name, confidence)
        print("VLM generation complete.")
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_label,
            'predicted_name': predicted_name,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'original_image': original_b64,
            'gradcam_image': gradcam_b64,
            'vlm_report': vlm_report
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vlm_loaded': vlm_model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Skin Cancer Classification Web Application")
    print("="*60)
    
    if load_models():
        print("\n[OK] Server starting...")
        print("[OK] Access the application at: http://localhost:5000")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("\n[ERROR] Failed to start server: Model loading failed")
        print("="*60 + "\n")
