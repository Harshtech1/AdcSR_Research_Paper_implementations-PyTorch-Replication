import os
import urllib.request
import zipfile
import shutil

def download_and_extract(url, zip_path, extract_dir):
    print(f"⬇️ Downloading {zip_path}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"📦 Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    os.remove(zip_path)

if __name__ == "__main__":
    # URLs for the standard DIV2K benchmark (Validation subset for fast download)
    hr_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    lr_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip"
    
    os.makedirs("tmp_data", exist_ok=True)
    
    # Download the High-Res and Low-Res zip archives
    download_and_extract(hr_url, "tmp_data/hr.zip", "tmp_data")
    download_and_extract(lr_url, "tmp_data/lr.zip", "tmp_data")
    
    # Paths to the extracted folders
    hr_source = "tmp_data/DIV2K_valid_HR"
    lr_source = "tmp_data/DIV2K_valid_LR_bicubic/X2"
    
    # Paths to your pipeline folders
    hr_target = os.path.expanduser("~/adcsr_pipeline/data/hr_images")
    lr_target = os.path.expanduser("~/adcsr_pipeline/data/lr_images")
    
    print("🔄 Renaming and moving files to your pipeline structure...")
    for f in os.listdir(hr_source):
        shutil.move(os.path.join(hr_source, f), os.path.join(hr_target, f))
        
    for f in os.listdir(lr_source):
        # DIV2K adds 'x2' to the low-res filename. We remove it so it matches the HR filename exactly.
        new_name = f.replace("x2.png", ".png")
        shutil.move(os.path.join(lr_source, f), os.path.join(lr_target, new_name))
        
    # Clean up the temporary downloads folder
    shutil.rmtree("tmp_data")
    
    print(f"✅ Success! {len(os.listdir(lr_target))} real image pairs added to your pipeline.")
