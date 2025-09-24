from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.report_generation_datasets import ReportGenerationDataset
from lavis.common.registry import registry

@registry.register_builder("report_generation")
class ReportGenerationBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReportGenerationDataset
    eval_dataset_cls = ReportGenerationDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/report_generation/defaults.yaml"
    }
