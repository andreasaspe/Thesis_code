import os
import subprocess
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

input_dir = "/storage/awias/s174197/data_RH/VertebraeSegmentation_newaffine/data_prep/img"
output_dir = "/storage/awias/s174197/data_RH/Predictions_newaffine/TotalSegmentator"
RH_segmentations_dir = "/storage/awias/s174197/data_RH/Predictions_newaffine/Segmentations_afterCCA"
task = "total"
# task = "total"
os.makedirs(output_dir, exist_ok=True)

all_series = [x.split("-")[0] for x in os.listdir(RH_segmentations_dir) if x.endswith(".nii.gz")]

print(all_series)

def segment_and_compress(series):
    try:
        filename = series + "-img.nii.gz"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, series + 'TS')
        nii_path = f"{output_path}.nii"
        gz_path = f"{output_path}.nii.gz"

        # Skip if compressed output already exists
        if os.path.exists(gz_path):
            print(f"Skipping {series}, compressed output file already exists.")
            return

        print(f"Segmenting: {filename}")
        subprocess.run([
            "TotalSegmentator",
            "-i", input_path,
            "-o", output_path,
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