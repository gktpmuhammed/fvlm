import os
import pandas as pd
from lavis.datasets.datasets.base_dataset import BaseDataset
from monai import transforms

class SpacingNormalization:
    """Custom transform to normalize spacing to reference spacing"""
    def __init__(self, ref_spacing=(1.0, 1.0, 3.0), debug=False):
        self.ref_spacing = ref_spacing
        self.debug = debug
    
    def __call__(self, data):
        affine = data["image_meta_dict"]["affine"]
        spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
        
        _, h, w, d = data["image"].shape
        scale = [spacing[i] / self.ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]
        
        if target_size != [h, w, d]:
            resize_transform = transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear")
            return resize_transform(data)
        else:
            return data

class ReportGenerationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.ann_paths = ann_paths
        
        # Enable test mode if we're in a test configuration
        self.test_mode = False  # Will be set externally for testing
        
        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
            SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=(112, 256, 352),
                mode="constant",
                constant_values=0
            ),
            transforms.CenterSpatialCropd(
                keys=["image"],
                roi_size=(112, 256, 352)
            ),
        ])

        self.annotation = self._load_annotations()
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        # Add organs attribute for compatibility with base task
        self.organs = ["chest"]  # Since this is CT chest imaging
    
    def _load_annotations(self):
        annotations = []
        
        ann_paths = self.ann_paths
        if not isinstance(ann_paths, list):
            ann_paths = [ann_paths]

        for ann_path in ann_paths:
            if 'train_reports' in os.path.basename(ann_path):
                split_folder = 'images/train'
            elif 'validation_reports' in os.path.basename(ann_path):
                split_folder = 'valid/images/valid'
            else:
                raise ValueError(f"Cannot determine split from annotation file path: {ann_path}")

            df = pd.read_csv(ann_path)
            df = df.dropna(subset=['Impressions_EN'])
            for index, row in df.iterrows():
                report = row["Impressions_EN"]
                volume_name = row["VolumeName"]
                
                # Extract the base name without extension to construct the nested path
                # e.g., train_1_a_1.nii.gz -> train_1/train_1_a/train_1_a_1.nii.gz
                base_name = volume_name.replace('.nii.gz', '')
                parts = base_name.split('_')
                if len(parts) >= 3:
                    # train_1_a_1 -> train_1/train_1_a/train_1_a_1.nii.gz
                    nested_path = f"{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}/{volume_name}"
                    image_path = os.path.join(self.vis_root, split_folder, nested_path)
                else:
                    # Fallback to original path
                    image_path = os.path.join(self.vis_root, split_folder, volume_name)
                
                if os.path.exists(image_path):
                    annotations.append({
                        "image_path": image_path,
                        "caption": report,
                        "image_id": row["VolumeName"] 
                    })
                    
                    # For testing: limit to small number of samples
                    # Check if we're in test mode by looking at environment variable
                    import os as os_module
                    if os_module.environ.get('FVLM_TEST_MODE', '').lower() == 'true':
                        if 'train_reports' in os.path.basename(ann_path) and len(annotations) >= 20:
                            print(f"Test mode: Limited train dataset to {len(annotations)} samples")
                            break
                        elif 'validation_reports' in os.path.basename(ann_path) and len(annotations) >= 5:
                            print(f"Test mode: Limited validation dataset to {len(annotations)} samples")
                            break
        return annotations

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        data = self.transform({"image": ann["image_path"]})
        image = data["image"]

        # For report generation, we need the raw text, not processed
        caption = ann["caption"]

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
