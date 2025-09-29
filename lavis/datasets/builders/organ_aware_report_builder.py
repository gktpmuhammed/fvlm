import os
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.organ_aware_report_datasets import OrganAwareReportDataset
from lavis.common.registry import registry
from lavis.common import utils

@registry.register_builder("organ_aware_report_generation")
class OrganAwareReportBuilder(BaseDatasetBuilder):
    train_dataset_cls = OrganAwareReportDataset
    eval_dataset_cls = OrganAwareReportDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/organ_aware_report_generation/defaults.yaml"
    }
    
    def build(self):
        """
        Create organ-aware datasets with JSON file paths
        """
        self.build_processors()

        build_info = self.config.build_info
        vis_info = build_info.get(self.data_type)

        # Get JSON file paths from config
        conc_info_path = build_info.get("conc_info_path")
        desc_info_path = build_info.get("desc_info_path")

        datasets = dict()
        
        # For organ-aware dataset, we create train and val splits based on JSON data
        # The dataset will automatically determine splits from JSON keys
        for split in ["train", "val"]:
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

            # visual data storage path
            vis_path = vis_info.storage
            if not os.path.isabs(vis_path):
                vis_path = utils.get_cache_path(vis_path)

            # create datasets with JSON paths
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                vis_root=vis_path,
                ann_paths=None,  # Not used in organ-aware dataset
                conc_info_path=conc_info_path,
                desc_info_path=desc_info_path,
                        max_samples=None  # Full dataset for comprehensive training
            )

        return datasets
