import SimpleITK as sitk
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

image_path = str(PROJECT_ROOT / 'data/preprocessed_samples_transpose/images/train_1_a_1_image.nii.gz')
mask_path = str(PROJECT_ROOT / 'data/preprocessed_samples_transpose/masks/train_1_a_1_mask.nii.gz')

if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
if not os.path.exists(mask_path):
    print(f"Mask file not found: {mask_path}")

try:
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    print("--- Image Properties ---")
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")
    print(f"Direction: {image.GetDirection()}")
    print(f"Pixel Type: {image.GetPixelIDValue()}")


    print("\n--- Mask Properties ---")
    print(f"Size: {mask.GetSize()}")
    print(f"Spacing: {mask.GetSpacing()}")
    print(f"Origin: {mask.GetOrigin()}")
    print(f"Direction: {mask.GetDirection()}")
    print(f"Pixel Type: {mask.GetPixelIDValue()}")

except Exception as e:
    print(f"An error occurred: {e}")

