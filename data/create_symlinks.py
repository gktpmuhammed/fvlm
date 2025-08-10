import os
import argparse
from pathlib import Path

def create_symlinks(source_dir, dest_dir, is_mask=False):
    """
    Creates symbolic links from a source directory to a destination directory,
    creating a nested structure from the filename.
    """
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_path = os.path.join(root, file)

            filename = file
            if is_mask:
                filename = filename.replace("_seg", "")

            base_name = filename.rsplit(".nii.gz", 1)[0]
            parts = base_name.split('_')

            # Create the nested directory structure from filename parts
            # e.g. train_1_a_1 -> train/train_1/train_1_a
            if len(parts) > 1:
                path_segments = []
                for i in range(1, len(parts)):
                    path_segments.append('_'.join(parts[:i]))
                dest_path_dir = os.path.join(dest_dir, *path_segments)
            else:
                dest_path_dir = dest_dir
            
            dest_path = os.path.join(dest_path_dir, filename)

            Path(dest_path_dir).mkdir(parents=True, exist_ok=True)
            
            if os.path.lexists(dest_path):
                # Using lexists to check for broken symlinks as well
                continue

            os.symlink(os.path.abspath(source_path), dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symbolic links for dataset.")
    parser.add_argument("--source_images", type=str, required=True, help="Source directory for images.")
    parser.add_argument("--source_masks", type=str, required=True, help="Source directory for masks.")
    parser.add_argument("--dest_root", type=str, required=True, help="Destination root directory.")
    
    args = parser.parse_args()

    # Create destination directories
    dest_images = os.path.join(args.dest_root, "images")
    dest_masks = os.path.join(args.dest_root, "masks")
    
    Path(dest_images).mkdir(parents=True, exist_ok=True)
    Path(dest_masks).mkdir(parents=True, exist_ok=True)

    create_symlinks(args.source_images, dest_images)
    create_symlinks(args.source_masks, dest_masks, is_mask=True)
