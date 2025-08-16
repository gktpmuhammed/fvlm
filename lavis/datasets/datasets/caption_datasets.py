"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
from collections import OrderedDict
from monai import transforms

from lavis.datasets.datasets.base_dataset import BaseDataset
import numpy as np
import random
import torch
import json
from pathlib import Path

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.vis_root = vis_root

        patient_paths = set()
        for root, _, files in os.walk(self.vis_root):
            if any(f.endswith('.nii.gz') for f in files):
                patient_paths.add(root)
        
        self.patient_paths = sorted(list(patient_paths))[:100]

        self.organs = [
            'face', 'brain', 'esophagus', 'trachea', 'lung', 'heart', 
            'kidney', 'stomach', 'liver', 'gallbladder', 'pancreas', 'spleen', 
            'colon', 'aorta', 'rib', 'humerus', 'scapula', 'clavicula', 
            'femur', 'hip', 'sacrum', 'gluteus', 'iliopsoas', 'autochthon'
        ]

        # Loading is handled in the visual processor (e.g., `fvlm_image_train`).
        # Avoid double applying LoadImaged here to prevent repeated I/O and logs.

        self.organ_ratios = {k: 1 for k in self.organs}

        desc_info = json.load(open('data/desc_info.json'))
        conc_info = json.load(open('data/conc_info.json'))

        all_info = {}
        for patient_path in self.patient_paths:
            patient = patient_path.split('/')[-1]

            all_info[patient] = {}
            for organ in self.organs:
                desc = ''
                if organ in desc_info.get(patient, {}):
                    desc += desc_info[patient][organ]
                    if not desc.endswith('.'):
                        desc += '.'

                conc = ''
                if organ in conc_info.get(patient, {}):
                    conc += conc_info[patient][organ]
                    if not conc.endswith('.'):
                        conc += '.'
                
                # Combine actual findings first, use default only if no real data
                if desc or conc:
                    input_text = conc + desc
                else:
                    input_text = f'{organ} shows no significant abnormalities.'

                input_text = input_text.replace('"', '')  
                input_text = input_text.replace('\'', '')  
                input_text = input_text.replace('(', '')  
                input_text = input_text.replace(')', '')

                all_info[patient][organ] = input_text
        self.annotation = all_info

        self.crop_size = (112, 256, 352)

    def __getitem__(self, index):
        max_retries = 10
        last_exception = None

        for _ in range(max_retries):
            try:
                patient_path = self.patient_paths[index]

                # Collect actual image files within this patient directory
                candidate_images = []
                for root, _, files in os.walk(patient_path):
                    for f in files:
                        if f.endswith('.nii.gz'):
                            candidate_images.append(os.path.join(root, f))

                if len(candidate_images) == 0:
                    raise FileNotFoundError(f"No .nii.gz files found under {patient_path}")

                img_path = random.choice(candidate_images)
                patient_id = patient_path.split('/')[-1]

                # Derive mask path from image path
                mask_path = img_path.replace('images', 'masks')

                # Let the visual processor handle initial MONAI transforms including LoadImaged
                data = self.vis_processor({'image': img_path, 'label': mask_path})

                # Enforce exact spatial size
                normalizer = transforms.Compose([
                    transforms.SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=self.crop_size,
                        method='end',
                        mode="constant",
                        constant_values=0
                    ),
                    transforms.CenterSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.crop_size
                    )
                ])
                data = normalizer(data)

                image = data['image']
                pul_seg = data['label'][0]

                text_input = self.annotation[patient_id]
                organ_abnormal_flags = torch.zeros(len(self.organs), dtype=bool)
                for i, organ in enumerate(self.organs):
                    if organ in text_input and not text_input[organ].startswith(f'{organ} shows no significant abnormalities.'):
                        organ_abnormal_flags[i] = True

                    if organ not in text_input:
                        text_input[organ] = f'{organ} shows no significant abnormalities.'

                return {
                    "image": image,
                    "seg": pul_seg,
                    "text_input": text_input,
                    "organ_abnormal_flags": organ_abnormal_flags
                }

            except Exception as e:
                last_exception = e
                print(e, patient_path)
                # Try a different patient on failure
                index = random.randint(0, len(self.patient_paths) - 1)
                continue

        # If all retries failed, raise the last exception for visibility
        raise RuntimeError(f"Failed to load sample after {max_retries} retries. Last error: {last_exception}")
