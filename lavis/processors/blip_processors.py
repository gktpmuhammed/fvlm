"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
from monai import transforms
import numpy as np

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Normalize

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        
        self.normalize = Normalize(mean, std)

@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.pre_caption(caption)
        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 80)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, captions):
        if isinstance(captions, str):
            captions = {"caption": captions}
            
        for (organ, caption) in captions.items():

            caption = re.sub(
                r"\s{2,}",
                " ",
                caption,
            )
            caption = caption.rstrip("\n")
            caption = caption.strip(" ")

            if caption and caption[-1] != '。':
                caption += '。'

            captions[organ] = caption

        return captions

@registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor):
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)

        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


@registry.register_processor("blip_image_train")
class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose([
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(112, 256, 352)
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.ToTensord(keys=["image", "label"])
        ])
    
    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip_image_eval")
class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


@registry.register_processor("fvlm_image_train")
class FVLMImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)
        
        # Load metadata for spacing correction
        import pandas as pd
        import os
        try:
            train_metadata = pd.read_csv('/home/muhammedg/fvlm/data/metadata/metadata/train_metadata.csv')
            try:
                val_metadata = pd.read_csv('/home/muhammedg/fvlm/data/metadata/metadata/validation_metadata.csv')
                self.metadata_df = pd.concat([train_metadata, val_metadata], ignore_index=True)
                print("✅ Loaded train + validation metadata for spacing correction")
            except:
                self.metadata_df = train_metadata  
                print("✅ Loaded train metadata for spacing correction (validation not found)")
        except Exception as e:
            print(f"⚠️ Could not load metadata: {e}")
            self.metadata_df = None

        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
            transforms.Lambdad(keys=["image"], func=self.fix_spacing_from_metadata),
            transforms.Lambdad(keys=["label"], func=self.merge_labels),
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 3.0), mode="trilinear"),
            transforms.Spacingd(keys=["label"], pixdim=(1.0, 1.0, 3.0), mode="nearest"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="label"),
            transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=(112, 256, 352),
                method='end',
                mode="constant",
                constant_values=0
            ),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(112, 256, 352)
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            self.ToRegularTensor(keys=["image", "label"])
        ])

    class ToRegularTensor:
        """Custom transform to convert MetaTensor to regular PyTorch tensor."""
        def __init__(self, keys):
            self.keys = keys
            
        def __call__(self, data):
            import torch
            import numpy as np
            
            for key in self.keys:
                if key in data:
                    item = data[key]
                    # Force conversion to regular PyTorch tensor with proper storage
                    try:
                        if hasattr(item, 'array'):
                            # MetaTensor case - create completely new tensor
                            array_data = np.array(item.array, copy=True)
                            data[key] = torch.from_numpy(array_data).clone()
                        elif hasattr(item, 'data'):
                            # Other MONAI tensor types
                            array_data = np.array(item.data, copy=True) 
                            data[key] = torch.from_numpy(array_data).clone()
                        elif hasattr(item, 'detach'):
                            # Already a tensor but might be MetaTensor
                            data[key] = torch.tensor(item.detach().cpu().numpy(), dtype=item.dtype)
                        else:
                            # Convert any other format
                            array_data = np.array(item, copy=True)
                            data[key] = torch.from_numpy(array_data).clone()
                    except Exception as e:
                        # Fallback: force numpy conversion
                        array_data = np.array(item, copy=True)
                        data[key] = torch.from_numpy(array_data).contiguous()
            return data

    def fix_spacing_from_metadata(self, image):
        """Fix spacing information from metadata before resampling."""
        import ast
        import os
        import numpy as np
        import torch
        
        if self.metadata_df is None:
            return image
            
        try:
            # Extract volume name from image path
            image_path = image.meta.get('filename_or_obj', '')
            if isinstance(image_path, str):
                volume_name = os.path.basename(image_path)
                
                # Find matching metadata row
                metadata_row = self.metadata_df[self.metadata_df['VolumeName'] == volume_name]
                
                if len(metadata_row) > 0:
                    metadata_row = metadata_row.iloc[0]
                    
                    # Get correct spacing from metadata
                    xy_spacing = ast.literal_eval(metadata_row['XYSpacing'])
                    z_spacing = metadata_row['ZSpacing']
                    correct_spacing = (xy_spacing[0], xy_spacing[1], z_spacing)
                    
                    # Update the affine matrix to use correct spacing
                    current_affine = image.affine
                    if isinstance(current_affine, torch.Tensor):
                        corrected_affine = current_affine.clone()
                    else:
                        corrected_affine = np.array(current_affine)
                    
                    # Set diagonal elements to correct spacing (with sign preservation)
                    corrected_affine[0, 0] = correct_spacing[0] * np.sign(float(current_affine[0, 0]))
                    corrected_affine[1, 1] = correct_spacing[1] * np.sign(float(current_affine[1, 1]))  
                    corrected_affine[2, 2] = correct_spacing[2] * np.sign(float(current_affine[2, 2]))
                    
                    # Simply update the metadata affine - let MONAI handle the rest
                    image.affine = corrected_affine
                    return image
                    
        except Exception as e:
            print(f"⚠️ Could not fix spacing for {volume_name if 'volume_name' in locals() else image_path}: {e}")
            
        return image

    def convert_to_tensor(self, data):
        """Convert MetaTensor to regular PyTorch tensor for DataLoader compatibility."""
        import torch
        import numpy as np
        
        # Force conversion to regular PyTorch tensor, stripping all MONAI metadata
        if hasattr(data, 'array'):
            # MetaTensor case
            return torch.tensor(np.array(data.array), dtype=data.dtype)
        elif hasattr(data, 'data'):
            # Other MONAI tensor types
            return torch.tensor(np.array(data.data), dtype=data.dtype)
        else:
            # Convert any tensor-like to pure PyTorch tensor
            return torch.tensor(np.array(data))

    def merge_labels(self, label):
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
            # Face/Head
            "skull": 0, "brain": 1,
            
            # Thoracic
            "esophagus": 2, "trachea": 3,
            "lung_upper_lobe_left": 4, "lung_lower_lobe_left": 4, "lung_upper_lobe_right": 4,
            "lung_middle_lobe_right": 4, "lung_lower_lobe_right": 4,
            "heart": 5, "atrial_appendage_left": 5,
            
            # Abdominal (removed: adrenal gland, small bowel, urinary bladder)
            "kidney_right": 6, "kidney_left": 6,
            "stomach": 7, "liver": 8, "gallbladder": 9, "pancreas": 10,
            "spleen": 11, "colon": 12,
            
            # Vascular (removed: inferior vena cava, portal vein, pulmonary artery, iliac vessels)
            "aorta": 13,
            
            # Ribs (grouped)
            "rib_left_1": 14, "rib_left_2": 14, "rib_left_3": 14, "rib_left_4": 14, "rib_left_5": 14,
            "rib_left_6": 14, "rib_left_7": 14, "rib_left_8": 14, "rib_left_9": 14, "rib_left_10": 14,
            "rib_left_11": 14, "rib_left_12": 14, "rib_right_1": 14, "rib_right_2": 14, "rib_right_3": 14,
            "rib_right_4": 14, "rib_right_5": 14, "rib_right_6": 14, "rib_right_7": 14, "rib_right_8": 14,
            "rib_right_9": 14, "rib_right_10": 14, "rib_right_11": 14, "rib_right_12": 14,
            
            # Bones  
            "humerus_left": 15, "humerus_right": 15,
            "scapula_left": 16, "scapula_right": 16,
            "clavicula_left": 17, "clavicula_right": 17,
            "femur_left": 18, "femur_right": 18,
            "hip_left": 19, "hip_right": 19,
            "sacrum": 20, "vertebrae_S1": 20,
            
            # Muscles
            "gluteus_maximus_left": 21, "gluteus_maximus_right": 21, "gluteus_medius_left": 21,
            "gluteus_medius_right": 21, "gluteus_minimus_left": 21, "gluteus_minimus_right": 21,
            "iliopsoas_left": 22, "iliopsoas_right": 22,
            "autochthon_left": 23, "autochthon_right": 23
        }
        
        fused_mask = np.zeros_like(label)
        for original_id, organ_name in class_map.items():
            if organ_name in merged_organ_id:
                merged_id = merged_organ_id[organ_name]
                fused_mask[label == original_id] = merged_id + 1
        return fused_mask

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        image_size = cfg.get("image_size", 384)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)
        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )
