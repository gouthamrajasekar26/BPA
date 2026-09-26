"""
MODULE 1: Automated Dataset Fetcher & Ingestion Engine
======================================================
Provides programmatic downloading, streaming, extraction, dataset splitting,
and statistics for the Attinger Bloodstain Pattern Dataset from Figshare.
Includes synthetic dataset generation capabilities for offline/demonstration runs.
"""

import os
import shutil
import random
import requests
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

FIGSHARE_ATTINGER_URL = "https://figshare.com/ndownloader/articles/7200500/versions/1"
DATASET_DIR = Path("data")
RAW_DIR = DATASET_DIR / "raw"
EXTRACTED_DIR = DATASET_DIR / "extracted"
SPLIT_DIR = DATASET_DIR / "split"

# Standardized Spatter Categories matching folder hierarchy
CATEGORIES = ['drip', 'beating', 'mvis', 'hvis']

CATEGORY_DESCRIPTIONS = {
    'drip': 'Passive Drip Spatter (< 1.5 m/s)',
    'beating': 'Blunt Force / Beating Spatter (1.5 - 7.5 m/s)',
    'mvis': 'Medium-Velocity Impact Spatter (1.5 - 7.5 m/s)',
    'hvis': 'High-Velocity Impact Spatter / Gunshot (> 30 m/s)'
}


def setup_directory_structure():
    """Creates directory hierarchy for raw data, extracted content, and split datasets."""
    dirs = [
        RAW_DIR,
        EXTRACTED_DIR,
        SPLIT_DIR / "train",
        SPLIT_DIR / "val",
        SPLIT_DIR / "test"
    ]
    for cat in CATEGORIES:
        for split in ['train', 'val', 'test']:
            dirs.append(SPLIT_DIR / split / cat)
            
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[Module 1] Directory structure initialized under: {DATASET_DIR.resolve()}")


def download_and_extract_attinger_dataset(url: str = FIGSHARE_ATTINGER_URL) -> bool:
    """
    Downloads the Attinger Bloodstain Pattern Dataset zip file from Figshare via HTTP streaming
    and extracts its contents into the raw data directory.
    """
    setup_directory_structure()
    zip_path = RAW_DIR / "attinger_dataset.zip"
    
    print(f"[Module 1] Fetching Attinger Dataset from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"[Module 1] Successfully downloaded zip archive to {zip_path}")
            
            # Extract archive
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(EXTRACTED_DIR)
            print(f"[Module 1] Extracted dataset archive to {EXTRACTED_DIR}")
            return True
        else:
            print(f"[Module 1] Warning: Figshare HTTP request returned status code {response.status_code}.")
            return False
    except Exception as e:
        print(f"[Module 1] Warning: Could not download Figshare dataset automatically: {e}")
        return False


def sort_and_stratify_dataset(train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Parses extracted filenames, categorizing 'drip', 'beating', 'mvis', 'hvis' samples,
    and performs a stratified 80/10/10 split across train, val, and test subdirectories.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    extracted_files = list(EXTRACTED_DIR.glob("**/*.*"))
    # Exclude directories
    extracted_files = [f for f in extracted_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]

    if not extracted_files:
        print("[Module 1] No raw extracted files found. Generating synthetic dataset for demonstration...")
        generate_synthetic_dataset()
        extracted_files = [f for f in EXTRACTED_DIR.glob("**/*.*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]

    categorized = {cat: [] for cat in CATEGORIES}
    
    for file_path in extracted_files:
        filename_lower = file_path.name.lower()
        if "drip" in filename_lower or "passive" in filename_lower:
            categorized['drip'].append(file_path)
        elif "beating" in filename_lower or "blunt" in filename_lower:
            categorized['beating'].append(file_path)
        elif "mvis" in filename_lower or "medium" in filename_lower:
            categorized['mvis'].append(file_path)
        elif "hvis" in filename_lower or "high" in filename_lower or "firearm" in filename_lower or "gunshot" in filename_lower:
            categorized['hvis'].append(file_path)
        else:
            # Hash fallback to guarantee allocation
            cat_idx = hash(file_path.name) % len(CATEGORIES)
            categorized[CATEGORIES[cat_idx]].append(file_path)

    print("[Module 1] Categorization statistics:")
    for cat, files in categorized.items():
        print(f"   - {cat.upper()} ({CATEGORY_DESCRIPTIONS[cat]}): {len(files)} files")
        random.shuffle(files)
        
        n_total = len(files)
        if n_total == 0:
            continue
            
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else 0
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        splits = [('train', train_files), ('val', val_files), ('test', test_files)]
        for split_name, split_list in splits:
            dest_dir = SPLIT_DIR / split_name / cat
            os.makedirs(dest_dir, exist_ok=True)
            for src in split_list:
                shutil.copy2(src, dest_dir / src.name)

    print("[Module 1] Stratified 80/10/10 dataset split complete.")


def generate_synthetic_dataset(num_samples_per_class: int = 25):
    """
    Generates synthetic bloodstain pattern images with realistic elliptical droplets,
    spines, and background wall texture for robust offline testing and validation.
    """
    setup_directory_structure()
    img_size = (600, 600)
    
    for cat in CATEGORIES:
        out_dir = EXTRACTED_DIR / cat
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            img = Image.new("RGB", img_size, color=(235, 230, 220)) # Wall plaster texture base
            draw = ImageDraw.Draw(img)
            
            # Add subtle wall noise
            noise = np.random.randint(-10, 10, (img_size[1], img_size[0], 3), dtype=np.int16)
            img_arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(img)
            
            blood_color = (130, 12, 15) # Dark crimson blood color
            
            if cat == 'drip':
                num_drops = random.randint(5, 12)
                for _ in range(num_drops):
                    cx, cy = random.randint(100, 500), random.randint(100, 500)
                    r = random.randint(15, 35) # Large circular drops
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=blood_color)
            elif cat == 'beating' or cat == 'mvis':
                num_drops = random.randint(20, 50)
                origin_x, origin_y = 300, 550
                for _ in range(num_drops):
                    angle_rad = np.radians(random.uniform(30, 150))
                    dist = random.uniform(50, 400)
                    cx = int(origin_x + dist * np.cos(angle_rad))
                    cy = int(origin_y - dist * np.sin(angle_rad))
                    
                    w = random.randint(4, 12)
                    h = int(w / max(0.2, np.sin(angle_rad)))
                    
                    # Draw ellipse droplet rotated toward origin
                    drop_img = Image.new("RGBA", (h * 2 + 10, h * 2 + 10), (0, 0, 0, 0))
                    drop_draw = ImageDraw.Draw(drop_img)
                    drop_draw.ellipse([h - w//2, h - h//2, h + w//2, h + h//2], fill=blood_color + (255,))
                    # Draw tail
                    drop_draw.line([h, h, h + w//3, h + int(h*0.6)], fill=blood_color + (255,), width=max(1, w//3))
                    
                    rot_deg = np.degrees(angle_rad) - 90
                    rotated = drop_img.rotate(rot_deg, expand=True)
                    img.paste(rotated, (cx - rotated.width//2, cy - rotated.height//2), rotated)
            elif cat == 'hvis':
                num_drops = random.randint(80, 180)
                origin_x, origin_y = 300, 300
                for _ in range(num_drops):
                    angle = random.uniform(0, 360)
                    dist = random.uniform(20, 250)
                    cx = int(origin_x + dist * np.cos(np.radians(angle)))
                    cy = int(origin_y + dist * np.sin(np.radians(angle)))
                    r = random.randint(1, 3) # Small mist-like droplets
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=blood_color)
            
            img.save(out_dir / f"{cat}_sample_{i+1:03d}.jpg")
            
    print(f"[Module 1] Synthetic dataset generated with {num_samples_per_class} images per category.")


def get_dataset_statistics() -> dict:
    """Returns directory counts, file paths, and class statistics for UI monitoring."""
    stats = {
        "raw_zip_exists": (RAW_DIR / "attinger_dataset.zip").exists(),
        "extracted_total": 0,
        "extracted_by_class": {},
        "split_counts": {"train": {}, "val": {}, "test": {}},
        "total_split_images": 0
    }
    
    for cat in CATEGORIES:
        extracted_cat_dir = EXTRACTED_DIR / cat
        cat_count = len(list(extracted_cat_dir.glob("*.*"))) if extracted_cat_dir.exists() else 0
        stats["extracted_by_class"][cat] = cat_count
        stats["extracted_total"] += cat_count
        
        for split in ['train', 'val', 'test']:
            split_cat_dir = SPLIT_DIR / split / cat
            c = len(list(split_cat_dir.glob("*.*"))) if split_cat_dir.exists() else 0
            stats["split_counts"][split][cat] = c
            stats["total_split_images"] += c
            
    return stats


if __name__ == "__main__":
    print("=== MODULE 1: Automated Dataset Fetcher & Ingestion ===")
    setup_directory_structure()
    success = download_and_extract_attinger_dataset()
    if not success:
        print("[Module 1] Falling back to synthetic dataset generation...")
        generate_synthetic_dataset(num_samples_per_class=20)
    sort_and_stratify_dataset()
    print("[Module 1] Execution complete! Dataset statistics:")
    print(get_dataset_statistics())
