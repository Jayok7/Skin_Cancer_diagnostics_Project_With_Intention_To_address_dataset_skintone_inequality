import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import argparse
from pathlib import Path
import random
import os

# Define VGG19 model for style transfer
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # We only need the features part of VGG19
        # '0': conv1_1, '5': conv2_1, '10': conv3_1, '19': conv4_1, '28': conv5_1
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name, max_size=400, shape=None):
    image = Image.open(image_name).convert('RGB')
    
    # Resize large images to speed up optimization
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
        
    if shape is not None:
        size = shape
        
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # Normalize with ImageNet mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

def reverse_transform(tensor):
    # Un-normalize back to 0-1 range for saving
    tensor = tensor.clone().detach()
    tensor = tensor.squeeze(0)
    for t, m, s in zip(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)

def calc_gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_transfer(content_path, style_path, output_path, device, model, steps=300):
    content_img = load_image(content_path).to(device)
    # Style image must match content image shape for ease of combining
    style_img = load_image(style_path, shape=[content_img.shape[2], content_img.shape[3]]).to(device)

    # We start optimizing from the content image (faster convergence and preserves structure better than noise)
    generated = content_img.clone().requires_grad_(True)
    
    # LBFGS is traditionally used for neural style transfer and converges fast
    optimizer = optim.LBFGS([generated])
    
    content_features = model(content_img)
    style_features = model(style_img)
    
    style_grams = [calc_gram_matrix(sf) for sf in style_features]

    # Hyperparameters
    # We want a high style weight to aggressively pull the skin tone/texture
    alpha = 1       # Content weight
    beta = 1000000  # Style weight

    step_i = [0]
    while step_i[0] <= steps:
        def closure():
            optimizer.zero_grad()
            generated_features = model(generated)
            
            # Content loss (using layer '19' which corresponds to index 3 in our chosen_features list)
            content_loss = torch.mean((generated_features[3] - content_features[3].detach()) ** 2)
            
            # Style loss
            style_loss = 0
            for gf, sg in zip(generated_features, style_grams):
                _, d, h, w = gf.size()
                gram = calc_gram_matrix(gf)
                style_loss += torch.mean((gram - sg.detach()) ** 2) / (d * h * w)
                
            total_loss = alpha * content_loss + beta * style_loss
            total_loss.backward()
            
            step_i[0] += 1
            if step_i[0] % 50 == 0:
                print(f"  Step {step_i[0]}: Total Loss {total_loss.item():.4f}")
                
            return total_loss
            
        optimizer.step(closure)
        
    final_img = reverse_transform(generated)
    save_image(final_img, output_path)

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content-dir", type=str, required=True, help="Directory with source lesion images")
    parser.add_argument("--style-dir", type=str, required=True, help="Directory with dark skin target images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save augmented images")
    parser.add_argument("--labels-csv", type=str, default="", help="Path to skin tone labels CSV (e.g., isic2019_skin_tone_labels.csv) to force Light/Medium content and Dark style.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of images to process")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps per image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained VGG
    model = VGG().to(device).eval()
    
    content_paths = list(Path(args.content_dir).glob("*.jpg"))
    style_paths = list(Path(args.style_dir).glob("*.jpg"))
    
    # Optional CSV filtering
    if args.labels_csv and os.path.exists(args.labels_csv):
        print(f"Loading skin tone labels from {args.labels_csv}...")
        light_medium_ids = set()
        dark_ids = set()
        import csv
        with open(args.labels_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row['image_id']
                label = row['predicted_label'].lower()
                if label in ['light', 'medium']:
                    light_medium_ids.add(img_id)
                elif label == 'dark':
                    dark_ids.add(img_id)
                    
        # Filter the loaded paths
        # We assume the file name starts with the image_id (e.g. ISIC_0000000.jpg)
        original_content_len = len(content_paths)
        content_paths = [p for p in content_paths if p.stem in light_medium_ids]
        
        original_style_len = len(style_paths)
        style_paths = [p for p in style_paths if p.stem in dark_ids]
        
        print(f"Filtered Content Images (Light/Medium): {len(content_paths)} / {original_content_len}")
        print(f"Filtered Style Images (Dark): {len(style_paths)} / {original_style_len}")
    
    if not content_paths or not style_paths:
        print("Missing images in content or style directories (or filtering removed them all).")
        return
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = min(args.num_samples, len(content_paths))
    random.shuffle(content_paths)
    
    for i in range(samples):
        c_path = content_paths[i]
        s_path = random.choice(style_paths)
        
        out_name = f"neural_style_{i}_{c_path.name}"
        out_path = output_dir / out_name
        
        print(f"\nProcessing {i+1}/{samples}: Content=[{c_path.name}] -> Style=[{s_path.name}]")
        style_transfer(c_path, s_path, out_path, device, model, steps=args.steps)
        
    print(f"\nFinished! Processed {samples} images to {output_dir}")

if __name__ == "__main__":
    main()
