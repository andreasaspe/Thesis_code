import os
import nibabel as nib
from scipy import ndimage
import numpy as np
from tqdm import tqdm

input_dir = "/storage/awias/s174197/data_Verse/Verse20_test_unpacked"
TS_dir = "/storage/awias/s174197/data_Verse/TotalSegmentator"
Verse_segmentations_dir = "/storage/awias/s174197/data_Verse/Predictions_GPU9/FULL_SEGMENTATIONS_batchnorm_afterCCA_evenbetterrotation/test_resampled"

all_series = [x for x in os.listdir(Verse_segmentations_dir) if x.endswith(".nii.gz")]

best_dice_Verse = []
best_dice_TS = []

for series in all_series:
    subject = series.split("_")[0]

    img_path = [x for x in os.listdir(input_dir) if x.startswith(subject) and x.endswith("img.nii.gz")][0]
    Verse_path = os.path.join(Verse_segmentations_dir, series)
    TS_path = os.path.join(TS_dir, subject + '_TS.nii.gz')
    GT_path = os.path.join(input_dir, img_path)

    if not os.path.exists(TS_path):
        # print(f"Missing TotalSegmentator output for {series}")
        continue

    # Load data
    Verse_nib = nib.load(Verse_path)
    Verse_data = Verse_nib.get_fdata()
    TS_nib = nib.load(TS_path)
    TS_data = TS_nib.get_fdata()
    GT_nib = nib.load(GT_path)
    GT_data = GT_nib.get_fdata()

    TS_data[(TS_data < 26) | (TS_data > 50)] = 0

    # Get unique labels in RH data (excluding 0/background)
    verse_labels = np.unique(Verse_data)
    verse_labels = verse_labels[verse_labels != 0]

    # Get unique labels in TS data (excluding 0/background)
    ts_labels = np.unique(TS_data)
    ts_labels = ts_labels[ts_labels != 0]

    # Get unique labels in GT data (excluding 0/background)
    gt_labels = np.unique(GT_data)
    gt_labels = gt_labels[gt_labels != 0]

    best_subject_dice_Verse = []  # Store Dice scores for GT labels
    best_subject_dice_TS = []  # Store Dice scores for TS labels

    # For each RH label, find the best matching GT label
    for verse_label in tqdm(verse_labels):
        rh_mask = Verse_data == verse_label
        best_overlap = 0
        best_gt_label = None

        for gt_label in gt_labels:
            gt_mask = GT_data == gt_label
            # Calculate Dice coefficient as overlap measure
            intersection = np.sum(rh_mask & gt_mask)
            dice = 2 * intersection / (np.sum(rh_mask) + np.sum(gt_mask))
            
            if dice > best_overlap:
                best_overlap = dice
                best_gt_label = gt_label

        if best_gt_label is not None:
            print(f"Subject: {subject}, best personal dice is: {best_overlap}")
            best_subject_dice_Verse.append(best_overlap)

    # For each TS label, find the best matching GT label
    for ts_label in tqdm(ts_labels):
        ts_mask = TS_data == ts_label
        best_overlap = 0
        best_gt_label = None

        for gt_label in gt_labels:
            gt_mask = GT_data == gt_label
            # Calculate Dice coefficient as overlap measure
            intersection = np.sum(ts_mask & gt_mask)
            dice = 2 * intersection / (np.sum(ts_mask) + np.sum(gt_mask))
            
            if dice > best_overlap:
                best_overlap = dice
                best_gt_label = gt_label

        if best_gt_label is not None:
            print(f"Subject: {subject}, best TS dice is: {best_overlap}")
            best_subject_dice_TS.append(best_overlap)

    best_subject_dice_Verse = np.array(best_subject_dice_Verse)
    best_subject_dice_TS = np.array(best_subject_dice_TS)

    print(f"Subject: {subject}, mean personal dice is: {np.mean(best_subject_dice_Verse)}")
    print(f"Subject: {subject}, mean TS dice is: {np.mean(best_subject_dice_TS)}")

    best_dice_Verse.append(np.mean(best_subject_dice_Verse))
    best_dice_TS.append(np.mean(best_subject_dice_TS))

print(f"Overall mean personal dice: {np.mean(best_dice_Verse)}")
print(f"Overall mean TS dice: {np.mean(best_dice_TS)}")