import SimpleITK as sitk
import os

image_path = '/mnt/nas/Data_WholeBody/CT-Rate/dataset/train_process/train_10_a_1.nii.gz'
mask_path = '/mnt/nas/Data_WholeBody/CT-Rate/dataset/train_TS/train_10_a_1_seg.nii.gz'

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

