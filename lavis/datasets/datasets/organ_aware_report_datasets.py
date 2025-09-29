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
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths=None, 
                 conc_info_path=None, desc_info_path=None, max_samples=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths or [])
        self.ann_paths = ann_paths  # Not used anymore, kept for compatibility
        
        # Define the 4 organs we're working with
        self.organs = ['lung', 'heart', 'esophagus', 'aorta']
        
        # Paths to decomposed report files
        self.conc_info_path = conc_info_path or "/home/muhammedg/test/fvlm/data/conc_info.json"
        self.desc_info_path = desc_info_path or "/home/muhammedg/test/fvlm/data/desc_info.json"
        self.max_samples = max_samples
        
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
        
        # Initialize image ID mapping
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    
    def merge_labels(self, label):
        """Merge anatomical structure labels to our 4 target organs"""
        class_map = {
            1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
            6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
            10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
            13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus",
            16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum",
            20: "colon", 21: "urinary_bladder", 22: "prostate", 23: "kidney_cyst_left",
            24: "kidney_cyst_right", 25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5",
            28: "vertebrae_L4", 29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
            32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9",
            36: "vertebrae_T8", 37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5",
            40: "vertebrae_T4", 41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
            44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4",
            48: "vertebrae_C3", 49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart",
            52: "aorta", 53: "pulmonary_vein", 54: "brachiocephalic_trunk",
            55: "subclavian_artery_right", 56: "subclavian_artery_left",
            57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
            59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
            61: "atrial_appendage_left", 62: "superior_vena_cava",
            63: "inferior_vena_cava", 64: "portal_vein_and_splenic_vein",
            65: "iliac_artery_left", 66: "iliac_artery_right", 67: "iliac_vena_left",
            68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
            71: "scapula_left", 72: "scapula_right", 73: "clavicula_left",
            74: "clavicula_right", 75: "femur_left", 76: "femur_right",
            77: "hip_left", 78: "hip_right", 79: "spinal_cord",
            80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
            82: "gluteus_medius_left", 83: "gluteus_medius_right",
            84: "gluteus_minimus_left", 85: "gluteus_minimus_right",
            86: "autochthon_left", 87: "autochthon_right", 88: "iliopsoas_left",
            89: "iliopsoas_right", 90: "brain", 91: "skull", 92: "rib_left_1",
            93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
            96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7",
            99: "rib_left_8", 100: "rib_left_9", 101: "rib_left_10",
            102: "rib_left_11", 103: "rib_left_12", 104: "rib_right_1",
            105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
            108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7",
            111: "rib_right_8", 112: "rib_right_9", 113: "rib_right_10",
            114: "rib_right_11", 115: "rib_right_12", 116: "sternum",
            117: "costal_cartilages"
        }
        
        merged_organ_id = {
            "lung_upper_lobe_left": 0,
            "lung_lower_lobe_left": 0,
            "lung_upper_lobe_right": 0,
            "lung_middle_lobe_right": 0,
            "lung_lower_lobe_right": 0,
            "heart": 1,
            "atrial_appendage_left": 1,
            "esophagus": 2,
            "aorta": 3,
        }
        
        import numpy as np
        fused_mask = np.zeros_like(label)
        for original_id, organ_name in class_map.items():
            if organ_name in merged_organ_id:
                merged_id = merged_organ_id[organ_name]
                fused_mask[label == original_id] = merged_id + 1
        return fused_mask
    
    def _load_json(self, file_path):
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return {}
    
    def _load_annotations(self):
        """Load annotations directly from JSON files"""
        annotations = []
        
        # Determine split based on which JSON files we have data for
        # We'll use the keys from conc_info.json as our primary source
        if not self.conc_info:
            print("Warning: No conclusion info loaded, dataset will be empty")
            return annotations
        
        # Get all case IDs from the JSON files
        case_ids = set(self.conc_info.keys())
        if self.desc_info:
            case_ids = case_ids.union(set(self.desc_info.keys()))
        
        for case_id in case_ids:
            # Parse case_id to determine paths (e.g., "train_1_a" -> train_1_a_1.nii.gz)
            parts = case_id.split('_')
            
            if len(parts) >= 3:
                # Construct volume name and paths
                volume_name = f"{case_id}_1.nii.gz"  # Assuming _1 suffix for volume files
                
                # Determine split folder based on case_id prefix
                if case_id.startswith('train'):
                    split_folder = 'images/train'
                elif case_id.startswith('valid'):
                    split_folder = 'valid/images/valid'
                else:
                    continue  # Skip unknown prefixes
                
                # Construct nested path: train_1_a_1 -> train_1/train_1_a/train_1_a_1.nii.gz
                nested_path = f"{parts[0]}_{parts[1]}/{case_id}/{volume_name}"
                image_path = os.path.join(self.vis_root, split_folder, nested_path)
                
                # Corresponding mask path
                mask_path = image_path.replace('images', 'masks')
                
                # Check if files exist
                if os.path.exists(image_path):
                    annotations.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "case_id": case_id,
                        "image_id": volume_name 
                    })
                    
                    # Apply max_samples limit if specified (takes priority over test mode)
                    if self.max_samples and len(annotations) >= self.max_samples:
                        print(f"Dataset limited to {self.max_samples} samples")
                        break
                    
                    # For testing: limit to small number of samples (only if max_samples not set)
                    import os as os_module
                    if not self.max_samples and os_module.environ.get('FVLM_TEST_MODE', '').lower() == 'true':
                        if case_id.startswith('train') and len([a for a in annotations if a['case_id'].startswith('train')]) >= 20:
                            print(f"Test mode: Limited train dataset to {len([a for a in annotations if a['case_id'].startswith('train')])} samples")
                            break
                        elif case_id.startswith('valid') and len([a for a in annotations if a['case_id'].startswith('valid')]) >= 5:
                            print(f"Test mode: Limited validation dataset to {len([a for a in annotations if a['case_id'].startswith('valid')])} samples")
                            break
        
        print(f"Loaded {len(annotations)} samples from JSON files")
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
    
    def _get_full_report(self, case_id):
        """Create full report by combining findings and impressions from JSON data"""
        # Get from decomposed JSON data
        findings_full = self.desc_info.get(case_id, {}).get("Findings", "")
        conc_full = self.conc_info.get(case_id, {}).get("Conclusion", "")
        
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
            
            # Apply organ ID mapping to merge anatomical structures to our 4 target organs
            seg_mask_np = seg_mask.numpy()
            merged_mask = self.merge_labels(seg_mask_np)
            seg_mask = torch.from_numpy(merged_mask).float()
            
            # Ensure image and mask have same shape
            assert image[0].shape == seg_mask.shape, f"Shape mismatch: image {image[0].shape}, mask {seg_mask.shape}"
            
        except Exception as e:
            print(f"Error loading {ann['image_path']}: {e}")
            # Return a random sample instead
            return self.__getitem__(random.randint(0, len(self.annotation) - 1))
        
        # Get organ-specific reports
        organ_reports = self._get_organ_reports(ann["case_id"])
        
        # Get full combined report
        full_report = self._get_full_report(ann["case_id"])
        
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
