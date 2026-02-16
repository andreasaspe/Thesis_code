import os
import subprocess
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

input_dir = "/storage/awias/s174197/data_Verse/Verse20_test_unpacked"
output_dir = "/storage/awias/s174197/data_Verse/TotalSegmentator"
Verse_segmentations_dir = "/storage/awias/s174197/data_Verse/Predictions_GPU9/FULL_SEGMENTATIONS_batchnorm_afterCCA_evenbetterrotation/test_resampled"
task = "total"
# task = "total"
os.makedirs(output_dir, exist_ok=True)

all_series = [x.split("_")[0] for x in os.listdir(Verse_segmentations_dir) if x.endswith(".nii.gz")]

print(all_series)

def segment_and_compress(series):
    try:
        filename_img = [x for x in os.listdir(input_dir) if x.startswith(series) and x.endswith("img.nii.gz")][0]
        input_path = os.path.join(input_dir, filename_img)
        output_path = os.path.join(output_dir, series + '_TS')
        nii_path = f"{output_path}.nii"
        gz_path = f"{output_path}.nii.gz"

        # Skip if compressed output already exists
        if os.path.exists(gz_path):
            print(f"Skipping {series}, compressed output file already exists.")
            return

        print(f"Segmenting: {filename_img}")
        subprocess.run([
            "TotalSegmentator",
            "-i", input_path,
            "-o", nii_path,
            "--task", task,
            "--ml"
        ], check=True)

        print("Compressing output file...")
        subprocess.run([
            "gzip",
            nii_path
        ], check=True)

        print(f"Success: {series}")
    
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subprocess failed for {series}: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error for {series}: {e}")



with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers to your CPU/GPU resources
    list(tqdm(executor.map(segment_and_compress, all_series), total=len(all_series)))