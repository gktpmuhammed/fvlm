from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.organ_aware_report_datasets import OrganAwareReportDataset
from lavis.common.registry import registry

@registry.register_builder("organ_aware_report_generation")
class OrganAwareReportBuilder(BaseDatasetBuilder):
    train_dataset_cls = OrganAwareReportDataset
    eval_dataset_cls = OrganAwareReportDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/organ_aware_report_generation/defaults.yaml"
    }
    
    def build_datasets(self):
        # Build datasets using the parent class method
        datasets = super().build_datasets()
        
        return datasets
