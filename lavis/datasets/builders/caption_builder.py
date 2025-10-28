"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import warnings
import lavis.common.utils as utils
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.caption_datasets import (
    CaptionDataset
)

from lavis.common.registry import registry


@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.
        Override to pass organs config to dataset.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        # Get organs and all_organs flag from config if available
        organs = getattr(self.config, 'organs', None)
        
        # Get all_organs flag from global config (set by train.py)
        from lavis.common.registry import registry
        global_config = registry.get("configuration")
        all_organs = getattr(global_config.config, 'all_organs', False) if global_config else False

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test", "infer_train"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            # Determine visual data storage path per split if provided
            if split == "val" and hasattr(build_info, "val_images"):
                vis_path = build_info.val_images.storage
            elif split == "test" and hasattr(build_info, "test_images"):
                vis_path = build_info.test_images.storage
            else:
                vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            
            # Pass organs and all_organs flag to dataset
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                organs=organs,
                all_organs=all_organs,
            )

        return datasets
