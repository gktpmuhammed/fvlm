import os
from pathlib import Path
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
from monai import transforms
import numpy as np
import torch

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
        # the conc and desc are processed here
        # caption = self.prompt + self.pre_caption(caption)
        caption = self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 80)    # 注意

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, captions):
        for (organ, caption) in captions.items():

            caption = re.sub(
                r"\s{2,}",
                " ",
                caption,
            )
            caption = caption.rstrip("\n")
            caption = caption.strip(" ")

            if caption[-1] != '。':
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

        # truncate question
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
        
        # NOTE: Using semi-preprocessed files from PROJECT_ROOT/data
        # These files have undergone some preprocessing but need additional steps
        # to match the reference preprocessing pipeline validated in detailed_comparison.py
        
        # Validated preprocessing pipeline that matches reference results
        self.transform = transforms.Compose([
            # transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),  # Keep metadata
            
            # Step 1: Apply label merging (using the limited 4-class merging)
            transforms.Lambdad(keys=["label"], func=self.merge_labels),
            
            # Step 2: Apply spacing normalization (validated to match reference)
            self.SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
            
            # Step 3: Apply remaining transforms in validated order
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            
            # Step 4: ROI cropping and padding (validated)
            self.Original_ROI_Crop_d(keys=["image", "label"]),
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
            # Training augmentations
            # transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            
            # Final tensor conversion
            self.ToRegularTensor(keys=["image", "label"])
        ])

    class SpacingNormalization:
        """Custom transform to normalize spacing to reference spacing"""
        def __init__(self, ref_spacing=(1.0, 1.0, 3.0), debug=False):
            self.ref_spacing = ref_spacing
            self.debug = debug
        
        def __call__(self, data):
            # Get original spacing from affine matrix
            affine = data["image_meta_dict"]["affine"]
            spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
            
            # Calculate scale and target size
            _, h, w, d = data["image"].shape
            scale = [spacing[i] / self.ref_spacing[i] for i in range(3)]
            target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]
            
            # Apply resizing if needed
            if target_size != [h, w, d]:
                resize_transform = transforms.Compose([
                    transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                    transforms.Resized(spatial_size=target_size, keys=["label"], mode="nearest"),
                ])
                result = resize_transform(data)
                
                # Manually update spacing metadata after resize
                for key in ["image", "label"]:
                    if key in result and hasattr(result[key], 'meta') and result[key].meta:
                        # Update pixdim with new spacing
                        if 'pixdim' in result[key].meta:
                            result[key].meta['pixdim'][1:4] = self.ref_spacing
                        
                        # Update affine matrix diagonal with new spacing  
                        if 'affine' in result[key].meta:
                            for i in range(3):
                                result[key].meta['affine'][i, i] = self.ref_spacing[i] * (1 if result[key].meta['affine'][i, i] > 0 else -1)
                
                return result
            else:
                return data

    class Original_ROI_Crop_d:
        """
        Custom dictionary-based transform to replicate the original pipeline's
        manual ROI cropping with fixed extensions.
        """
        def __init__(self, keys, extend_d=5, extend_hw=20):
            self.keys = keys
            self.extend_d = extend_d
            self.extend_hw = extend_hw

        def __call__(self, data):
            d = data.copy()
            image = d["image"]
            label = d["label"]

            # Ensure label is integer type for nonzero operation
            if not isinstance(label, torch.Tensor):
                label = torch.as_tensor(label)
            
            if label.dtype not in (torch.int, torch.long, torch.int8, torch.int16):
                 label = label.long()

            if torch.sum(label) > 0:
                # Use torch.nonzero to find bounding box
                roi_coords_tuple = torch.nonzero(label[0], as_tuple=True)
                
                min_dhw = torch.tensor([torch.min(coords) for coords in roi_coords_tuple])
                max_dhw = torch.tensor([torch.max(coords) for coords in roi_coords_tuple])

                min_dhw = torch.maximum(
                    min_dhw - torch.tensor([self.extend_d, self.extend_hw, self.extend_hw]),
                    torch.tensor([0, 0, 0]),
                )
                max_dhw = torch.minimum(
                    max_dhw + torch.tensor([self.extend_d, self.extend_hw, self.extend_hw]),
                    torch.tensor([image.shape[1], image.shape[2], image.shape[3]]),
                )

                for key in self.keys:
                    d[key] = d[key][
                        :, min_dhw[0] : max_dhw[0], min_dhw[1] : max_dhw[1], min_dhw[2] : max_dhw[2]
                    ]
            
            return d

    class ToRegularTensor:
        """Custom transform to convert MetaTensor to regular PyTorch tensor while preserving metadata."""
        def __init__(self, keys):
            self.keys = keys
            
        def __call__(self, data):
            import torch
            import numpy as np
            
            for key in self.keys:
                if key in data:
                    item = data[key]
                    
                    # Store metadata before conversion
                    original_meta = None
                    if hasattr(item, 'meta') and item.meta is not None:
                        original_meta = dict(item.meta)  # Make a copy
                    
                    # Force conversion to regular PyTorch tensor with proper storage
                    try:
                        if hasattr(item, 'array'):
                            # MetaTensor case - create completely new tensor
                            array_data = np.array(item.array, copy=True)
                            new_tensor = torch.from_numpy(array_data).clone()
                        elif hasattr(item, 'data'):
                            # Other MONAI tensor types
                            array_data = np.array(item.data, copy=True) 
                            new_tensor = torch.from_numpy(array_data).clone()
                        elif hasattr(item, 'detach'):
                            # Already a tensor but might be MetaTensor
                            new_tensor = torch.tensor(item.detach().cpu().numpy(), dtype=item.dtype)
                        else:
                            # Convert any other format
                            array_data = np.array(item, copy=True)
                            new_tensor = torch.from_numpy(array_data).clone()
                            
                        # Restore metadata if it existed
                        if original_meta is not None:
                            # Create a new MetaTensor with preserved metadata
                            from monai.data import MetaTensor
                            data[key] = MetaTensor(new_tensor, meta=original_meta)
                        else:
                            data[key] = new_tensor
                            
                    except Exception as e:
                        # Fallback: force numpy conversion
                        array_data = np.array(item, copy=True)
                        data[key] = torch.from_numpy(array_data).contiguous()
            return data





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
        
        # merged_organ_id = {
        #     "lung_upper_lobe_left": 0,
        #     "lung_lower_lobe_left": 0,
        #     "lung_upper_lobe_right": 0,
        #     "lung_middle_lobe_right": 0,
        #     "lung_lower_lobe_right": 0,
        #     "heart": 1,
        #     "atrial_appendage_left": 1,
        #     "esophagus": 2,
        #     "aorta": 3,
        # }
        
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
