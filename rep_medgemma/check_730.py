import os
from pathlib import Path
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

path = str(PROJECT_ROOT / 'data_sym/valid/masks/valid/valid_730/valid_730_a/valid_730_a_1.nii.gz')
img = nib.load(path)
data = img.get_fdata() # (H, W, D) usually (e.g. 512, 512, 300)

print(f"Original Shape: {data.shape}")
u = np.unique(data).astype(int)
print(f"Unique IDs (Pre-Crop): {u}")
has_kidney = (2 in u) or (3 in u)
print(f"Has Kidney (Original)? {has_kidney}")

# Simulate Center Crop (approx)
# Target: (112, 256, 352) -> (D, H, W)
# But here data is (H, W, D)
# So target (H=256, W=352, D=112)??
# Wait, Transposed(0, 3, 2, 1) implies D comes to index 1.
# So transforms operate on (C, D, H, W).
# So spatial dimensions are (D, H, W).
# So crop is (112, 256, 352).
# This means crop D=112, H=256, W=352.

# Data is (H, W, D).
H, W, D = data.shape
print(f"H={H}, W={W}, D={D}")

# Crop D (axis 2) to 112
d_start = max(0, (D - 112) // 2)
d_end = min(D, d_start + 112)

# Crop H (axis 0) to 256
# Wait, target H is 256? Or W?
# Transposed indices (0, 3, 2, 1) -> Logic:
# Input (C, H, W, D) -> (C, D, H, W)
# Crop refers to spatial axes of the transform output? Or input?
# If `CenterSpatialCropd` operates on spatial axes...
# If `EnsureChannelFirstd` makes (C, H, W, D).
# `Transposed` makes (C, D, H, W).
# Then `CenterSpatialCropd` (roi_size=(112, 256, 352)) operates on (D, H, W).
# So D=112, H=256, W=352?
# H=256 from 512? That's half the image!
# W=352 from 512?
# So let's crop H to 256 (center), W to 352 (center).

h_start = max(0, (H - 256) // 2)
h_end = min(H, h_start + 256)

w_start = max(0, (W - 352) // 2)
w_end = min(W, w_start + 352)

print(f"Simulated Crop: H[{h_start}:{h_end}], W[{w_start}:{w_end}], D[{d_start}:{d_end}]")

cropped = data[h_start:h_end, w_start:w_end, d_start:d_end]
u_crop = np.unique(cropped).astype(int)
has_kidney_crop = (2 in u_crop) or (3 in u_crop)
print(f"Has Kidney (Cropped)? {has_kidney_crop}")
print(f"Unique IDs (Cropped): {u_crop}")
