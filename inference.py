import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from tqdm import tqdm

from train_stage2 import LightningAdcSRStage2Student

def run_batch_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing AdcSR Batch Inference Engine on {device}...")

    # 1. Find the latest Stage 2 checkpoint automatically
    checkpoint_dir = "lightning_logs/"
    if not os.path.exists(checkpoint_dir):
        print("⚠️ 'lightning_logs' not found.")
        return
        
    version_folders = sorted(glob.glob(os.path.join(checkpoint_dir, "version_*")), key=os.path.getmtime, reverse=True)
    latest_checkpoints = glob.glob(os.path.join(version_folders[0], "checkpoints", "*.ckpt"))
    
    if not latest_checkpoints:
        print("⚠️ No checkpoint (.ckpt) files found.")
        return
        
    checkpoint_path = latest_checkpoints[0]
    print(f"📦 Loading trained weights from: {checkpoint_path}")

    # 2. Load the Model
    model = LightningAdcSRStage2Student.load_from_checkpoint(checkpoint_path)
    model.eval()  
    model.to(device)

    # 3. Setup Folders
    input_dir = "data/lr_images"
    output_dir = "data/generated_results"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"✨ Starting batch generation for {len(image_files)} images...")

    # 4. The Batch Loop
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Upscaling Images"):
            filename = os.path.basename(img_path)
            
            # Read & Transform
            lr_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(lr_img).unsqueeze(0).to(device)
            
            # Forward Pass
            _, hr_tensor = model(input_tensor)
            
            # De-normalize and Save
            hr_tensor = (hr_tensor.squeeze(0).clamp(-1, 1) + 1) / 2.0
            hr_img = transforms.ToPILImage()(hr_tensor.cpu())
            hr_img.save(os.path.join(output_dir, filename))

    print(f"🎉 Success! All {len(image_files)} images saved to {output_dir}")

if __name__ == "__main__":
    run_batch_inference()
