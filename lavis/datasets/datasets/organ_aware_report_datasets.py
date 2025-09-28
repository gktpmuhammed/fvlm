import os
import json
import pandas as pd
import torch
import random
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

class OrganAwareReportDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, 
                 conc_info_path=None, desc_info_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.ann_paths = ann_paths
        
        # Define the 4 organs we're working with
        self.organs = ['lung', 'heart', 'esophagus', 'aorta']
        
        # Paths to decomposed report files
        self.conc_info_path = conc_info_path or "/home/muhammedg/test/fvlm/data/conc_info.json"
        self.desc_info_path = desc_info_path or "/home/muhammedg/test/fvlm/data/desc_info.json"
        
        # Load decomposed reports
        self.conc_info = self._load_json(self.conc_info_path)
        self.desc_info = self._load_json(self.desc_info_path)
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
            SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=(112, 256, 352),
                mode="constant",
                constant_values=0
            ),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(112, 256, 352)
            ),
        ])
        
        # Load annotations and create image ID mapping
        self.annotation = self._load_annotations()
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    
    def _load_json(self, file_path):
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return {}
    
    def _load_annotations(self):
        """Load annotations from CSV files"""
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
                volume_name = row["VolumeName"]
                
                # Extract the base name without extension to construct the nested path
                base_name = volume_name.replace('.nii.gz', '')
                parts = base_name.split('_')
                
                if len(parts) >= 3:
                    # train_1_a_1 -> train_1/train_1_a/train_1_a_1.nii.gz
                    nested_path = f"{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}/{volume_name}"
                    image_path = os.path.join(self.vis_root, split_folder, nested_path)
                    
                    # Corresponding mask path
                    mask_path = image_path.replace('images', 'masks')
                    
                    # Case ID for decomposed reports (e.g., "train_1_a")
                    case_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    
                else:
                    # Fallback to original path
                    image_path = os.path.join(self.vis_root, split_folder, volume_name)
                    mask_path = image_path.replace('images', 'masks')
                    case_id = base_name
                
                if os.path.exists(image_path):
                    annotations.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "case_id": case_id,
                        "findings": row.get("Findings_EN", ""),
                        "impressions": row.get("Impressions_EN", ""),
                        "image_id": volume_name 
                    })
                    
                    # For testing: limit to small number of samples
                    import os as os_module
                    if os_module.environ.get('FVLM_TEST_MODE', '').lower() == 'true':
                        if 'train_reports' in os.path.basename(ann_path) and len(annotations) >= 20:
                            print(f"🧪 Test mode: Limited train dataset to {len(annotations)} samples")
                            break
                        elif 'validation_reports' in os.path.basename(ann_path) and len(annotations) >= 5:
                            print(f"🧪 Test mode: Limited validation dataset to {len(annotations)} samples")
                            break
        
        return annotations

    def _get_organ_reports(self, case_id):
        """Extract organ-specific reports for a case"""
        organ_reports = {}
        
        # Get organ-specific texts from both findings and conclusions
        findings_data = self.desc_info.get(case_id, {})
        conc_data = self.conc_info.get(case_id, {})
        
        for organ in self.organs:
            # Combine findings and conclusion for each organ
            findings_text = findings_data.get(organ, "")
            conc_text = conc_data.get(organ, "")
            
            # Combine with space if both exist
            if findings_text and conc_text:
                combined_text = f"{findings_text} {conc_text}"
            elif findings_text:
                combined_text = findings_text
            elif conc_text:
                combined_text = conc_text
            else:
                # Default text for missing organs
                combined_text = f"{organ} shows no significant abnormalities."
            
            organ_reports[organ] = combined_text.strip()
        
        return organ_reports
    
    def _get_full_report(self, case_id, findings, impressions):
        """Create full report by combining findings and impressions"""
        # Try to get from decomposed data first
        findings_full = self.desc_info.get(case_id, {}).get("Findings", "")
        conc_full = self.conc_info.get(case_id, {}).get("Conclusion", "")
        
        # Fallback to CSV data if decomposed data not available
        if not findings_full:
            findings_full = findings or ""
        if not conc_full:
            conc_full = impressions or ""
        
        # Combine findings and impressions
        if findings_full and conc_full:
            full_report = f"{findings_full} {conc_full}"
        elif findings_full:
            full_report = findings_full
        elif conc_full:
            full_report = conc_full
        else:
            full_report = "No significant abnormalities detected."
        
        return full_report.strip()
    
    def _get_organ_abnormal_flags(self, organ_reports):
        """Determine which organs have abnormalities based on text content"""
        organ_abnormal_flags = torch.zeros(len(self.organs), dtype=bool)
        
        for i, organ in enumerate(self.organs):
            organ_text = organ_reports[organ].lower()
            # Check if the text indicates abnormalities (not just "no abnormalities")
            if not ("no significant abnormalities" in organ_text or 
                   "no abnormality" in organ_text or
                   "normal" in organ_text):
                organ_abnormal_flags[i] = True
        
        return organ_abnormal_flags

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        # Load image and segmentation mask
        try:
            data = self.transform({
                "image": ann["image_path"],
                "label": ann["mask_path"]
            })
            image = data["image"]
            seg_mask = data["label"][0].as_tensor()  # Remove channel dimension
            
            # Ensure image and mask have same shape
            assert image[0].shape == seg_mask.shape, f"Shape mismatch: image {image[0].shape}, mask {seg_mask.shape}"
            
        except Exception as e:
            print(f"Error loading {ann['image_path']}: {e}")
            # Return a random sample instead
            return self.__getitem__(random.randint(0, len(self.annotation) - 1))
        
        # Get organ-specific reports
        organ_reports = self._get_organ_reports(ann["case_id"])
        
        # Get full combined report
        full_report = self._get_full_report(ann["case_id"], ann["findings"], ann["impressions"])
        
        # Get organ abnormality flags
        organ_abnormal_flags = self._get_organ_abnormal_flags(organ_reports)
        
        return {
            "image": image,
            "seg": seg_mask,
            "text_input": full_report,
            "organ_reports": organ_reports,
            "organ_abnormal_flags": organ_abnormal_flags,
            "mode": "generation",
            "image_id": self.img_ids[ann["image_id"]],
            "case_id": ann["case_id"]
        }
