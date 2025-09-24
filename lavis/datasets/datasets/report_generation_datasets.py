import os
from PIL import Image
import pandas as pd
from lavis.datasets.datasets.base_dataset import BaseDataset

class ReportGenerationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.ann_paths = ann_paths
        self.annotation = self._load_annotations()
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    
    def _load_annotations(self):
        annotations = []
        
        ann_paths = self.ann_paths
        if not isinstance(ann_paths, list):
            ann_paths = [ann_paths]

        for ann_path in ann_paths:
            if 'train_reports' in os.path.basename(ann_path):
                split_folder = 'images/train'
            elif 'validation_reports' in os.path.basename(ann_path):
                split_folder = 'valid/images'
            else:
                raise ValueError(f"Cannot determine split from annotation file path: {ann_path}")

            df = pd.read_csv(ann_path)
            df = df.dropna(subset=['Impressions_EN'])
            for index, row in df.iterrows():
                report = row["Impressions_EN"]
                image_path = os.path.join(self.vis_root, split_folder, row["VolumeName"])
                
                if os.path.exists(image_path):
                    annotations.append({
                        "image_path": image_path,
                        "caption": report,
                        "image_id": row["VolumeName"] 
                    })
        return annotations

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        # This is a placeholder for how you might load your 3D images
        # For now, we'll just return a blank image to get the structure in place
        image = Image.new('RGB', (256, 256)) 

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
