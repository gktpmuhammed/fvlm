
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration: Mapping Organ Index to Disease Names (Column Headers)
# Organ Order in train.py: 
# ['lung', 'heart', 'aorta', 'esophagus', 'trachea', 'rib', 
#  'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney']

from pydantic import BaseModel, Field

# Configuration: Mapping Organ Index to Disease Names (Column Headers)
# Organ Order in train.py: 
# ['lung', 'heart', 'aorta', 'esophagus', 'trachea', 'rib', 
#  'liver', 'gallbladder', 'stomach', 'pancreas', 'spleen', 'kidney']

ORGAN_INDEX_MAP = {
    'lung': 0,
    'heart': 1,
    'aorta': 2,
    'esophagus': 3,
    'trachea': 4,
    'rib': 5,
    'liver': 6,
    'gallbladder': 7,
    'stomach': 8,
    'pancreas': 9,
    'spleen': 10,
    'kidney': 11
}

class EsophagusDiseases(BaseModel):
    hiatal_hernia: int = Field(..., description="Hiatal hernia")
    varicose_veins: int = Field(..., description="Esophageal varices")

class GallbladderDiseases(BaseModel):
    cholecystitis: int = Field(..., description="Cholecystitis")
    gallstone: int = Field(..., description="Gallstones, cholelithiasis")
    adenomyomatosis: int = Field(..., description="Adenomyomatosis")

class HeartDiseases(BaseModel):
    cardiomegaly: int = Field(..., description="Cardiomegaly, enlarged heart")
    pericardial_effusion: int = Field(..., description="Pericardial effusion")

class KidneyDiseases(BaseModel):
    atrophy: int = Field(..., description="Renal atrophy")
    cyst: int = Field(..., description="Renal cyst")
    hydronephrosis: int = Field(..., description="Hydronephrosis")
    calculi: int = Field(..., description="Renal calculi, stones")

class LiverDiseases(BaseModel):
    steatosis: int = Field(..., description="Steatosis, fatty liver")
    glissons_capsule_effusion: int = Field(..., description="Perihepatic fluid/effusion")
    metastasis: int = Field(..., description="Liver metastasis")
    intrahepatic_duct_dilatation: int = Field(..., description="Intrahepatic bile duct dilatation")
    cancer: int = Field(..., description="Primary liver cancer/HCC")
    cyst: int = Field(..., description="Liver cyst")
    abscess: int = Field(..., description="Liver abscess")
    cirrhosis: int = Field(..., description="Cirrhosis")

class LungDiseases(BaseModel):
    atelectasis: int = Field(..., description="Atelectasis")
    bronchiectasis: int = Field(..., description="Bronchiectasis")
    emphysema: int = Field(..., description="Emphysema")
    pneumonia: int = Field(..., description="Pneumonia, consolidation, infiltrate")
    pleural_effusion: int = Field(..., description="Pleural effusion")

class PancreasDiseases(BaseModel):
    pancreatic_cancer: int = Field(..., description="Pancreatic cancer/mass")
    atrophy: int = Field(..., description="Pancreatic atrophy")
    pancreatitis: int = Field(..., description="Pancreatitis")
    duct_dilatation: int = Field(..., description="Pancreatic duct dilatation")
    steatosis: int = Field(..., description="Lipomatosis/steatosis of pancreas")

class SpleenDiseases(BaseModel):
    hemangioma: int = Field(..., description="Hemangioma")
    infarction: int = Field(..., description="Splenic infarction")
    splenomegaly: int = Field(..., description="Splenomegaly")

class StomachDiseases(BaseModel):
    wall_thickening: int = Field(..., description="Gastric wall thickening")
    cancer: int = Field(..., description="Stomach/Gastric cancer")

# Mapping Keys
ORGAN_TO_SCHEMA = {
    "esophagus": EsophagusDiseases,
    "gallbladder": GallbladderDiseases,
    "heart": HeartDiseases,
    "kidney": KidneyDiseases,
    "liver": LiverDiseases,
    "lung": LungDiseases,
    "pancreas": PancreasDiseases,
    "spleen": SpleenDiseases,
    "stomach": StomachDiseases,
}

# Dynamically generate DISEASE_CONFIG
DISEASE_CONFIG = {}
for organ, schema in ORGAN_TO_SCHEMA.items():
    DISEASE_CONFIG[organ] = list(schema.model_fields.keys())

class DiseaseAuxiliaryLoss(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.organ_classifiers = nn.ModuleDict()
        self.disease_map = DISEASE_CONFIG
        self.organ_indices = ORGAN_INDEX_MAP
        
        # Create a classifier head for each organ that has associated diseases
        for organ, diseases in self.disease_map.items():
            num_classes = len(diseases)
            self.organ_classifiers[organ] = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim // 2, num_classes)
            )

    def forward(self, organ_embeddings, disease_labels):
        """
        organ_embeddings: (Batch, Num_Organs, Embed_Dim)
        disease_labels: Dict[str, Tensor(Batch, Num_Diseases_For_Organ)] 
                        OR a consolidated Tensor if handled upstream.
                        
        For simplicity, we assume `disease_labels` is a dictionary keyed by organ name,
        containing 0/1 tensors for that organ's diseases.
        """
        total_loss = 0.0
        details = {}
        
        for organ, classifier in self.organ_classifiers.items():
            if organ not in disease_labels:
                continue
                
            idx = self.organ_indices[organ]
            # Select the specific organ embedding
            # Shape: (Batch, Embed_Dim)
            specific_organ_emb = organ_embeddings[:, idx, :] 
            
            # Predict
            logits = classifier(specific_organ_emb)
            
            # Get Targets
            targets = disease_labels[organ].to(logits.device).float()
            
            # Compute Loss
            # Using BCEWithLogits for multi-label classification
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            
            total_loss += loss
            details[f"loss_{organ}"] = loss.item()
            
        return total_loss, details
