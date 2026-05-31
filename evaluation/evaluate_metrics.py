import os
import glob
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

def calculate_metrics(generated_dir, ground_truth_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Initializing Evaluation Metrics Engine on {device}...")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net='vgg').to(device)

    generated_files = sorted(glob.glob(os.path.join(generated_dir, "*.png")))
    
    if len(generated_files) == 0:
        print(f"⚠️ No images found in {generated_dir}. Please run inference first!")
        return

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    # FIXED: Ensure Ground Truth matches the 512x512 Generated Output
    to_tensor = ToTensor()
    gt_transform = Compose([
        Resize((512, 512)),
        ToTensor()
    ])

    print("🔍 Scanning image pairs...")
    for gen_path in generated_files:
        filename = os.path.basename(gen_path)
        gt_path = os.path.join(ground_truth_dir, filename)

        if not os.path.exists(gt_path):
            continue

        # Load images and match dimensions
        gen_img = to_tensor(Image.open(gen_path).convert('RGB')).unsqueeze(0).to(device)
        gt_img = gt_transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            total_psnr += psnr_metric(gen_img, gt_img).item()
            total_ssim += ssim_metric(gen_img, gt_img).item()
            
            gen_img_lpips = (gen_img * 2) - 1
            gt_img_lpips = (gt_img * 2) - 1
            total_lpips += lpips_metric(gen_img_lpips, gt_img_lpips).item()
            
        count += 1

    if count == 0:
        print("⚠️ No matching ground truth files found.")
        return

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count

    print("\n" + "="*40)
    print("🏆 FINAL EVALUATION METRICS 🏆")
    print("="*40)
    print(f"Total Images Evaluated : {count}")
    print(f"PSNR  (Higher is better): {avg_psnr:.2f} dB")
    print(f"SSIM  (Higher is better): {avg_ssim:.4f}")
    print(f"LPIPS (Lower is better) : {avg_lpips:.4f}")
    print("="*40)

if __name__ == "__main__":
    GENERATED_FOLDER = "data/generated_results"
    GROUND_TRUTH_FOLDER = "data/hr_images"
    os.makedirs(GENERATED_FOLDER, exist_ok=True)
    calculate_metrics(GENERATED_FOLDER, GROUND_TRUTH_FOLDER)
