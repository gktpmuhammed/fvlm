import os
import argparse
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from monai import transforms

from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.common.dist_utils import get_rank, init_distributed_mode


def collate_fn(batch):
    return batch[0]

@torch.no_grad()
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def merge_labels(label):
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
    
    fused_mask = torch.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask

class DataFolder(Dataset):
    def __init__(self, vis_root='data/valid/images'):
        super().__init__()

        img_paths = []
        for root, _, files in os.walk(vis_root):
            for file in files:
                if file.endswith('.nii.gz'):
                    img_paths.append(os.path.join(root, file))
        self.img_paths = img_paths[:100]

        if not self.img_paths:
            raise FileNotFoundError(f"No .nii.gz files found in {vis_root}")

        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
            transforms.Lambdad(keys=["label"], func=merge_labels),
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
        
        if dist.is_initialized():
            self.img_paths = self.img_paths[dist.get_rank()::dist.get_world_size()]
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = image_path.replace('images', 'masks')
        
        file_name = image_path.split('/')[-1]
        input_path = {'image': image_path, 'label': mask_path}

        data = self.transform(input_path)
        
        return {
            "image": data['image'],
            "file_name": file_name
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Report Generation Evaluation")
    parser.add_argument('--vis_root', type=str, default='data/valid/images', help='The path to the visual root directory.')
    parser.add_argument('--ckpt_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth', help='The path to the checkpoint file.')
    parser.add_argument("--cfg-path", required=False, default='lavis/projects/blip/train/pretrain_ct.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # Generation parameters
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--min_length", type=int, default=10)


    args = parser.parse_args()
    return args

@torch.inference_mode()
def evaluate():
    args = parse_args()

    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)

    datafolder = DataFolder(args.vis_root)
    dataloader = DataLoader(
        datafolder,
        batch_size=1, # generate one by one
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    ckpt_path = args.ckpt_path
    
    ckpt = torch.load(
        ckpt_path, map_location='cpu'
    )
    
    model.load_state_dict(ckpt['model'], strict=False)

    rank = get_rank()
    torch.cuda.set_device(rank)

    model.eval()
    model.cuda()

    results = []
    
    for i, samples in enumerate(tqdm(dataloader, desc='Generating Reports')):
        samples["image"] = samples["image"].cuda()
        
        generated_text = model.generate(
            samples,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length
        )

        results.append({"file_name": samples["file_name"], "report": generated_text[0]})
    
    if dist.is_initialized():
        dist.barrier()
        gathered_results = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_results, results)
        results = [item for sublist in gathered_results for item in sublist]

    if rank == 0:
        save_path = '_'.join(ckpt_path.replace('.pth', '').split('/')[1:])
        results_file = f'report_generations/{save_path}_reports.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'Saved generated reports to {results_file}')

if __name__ == '__main__':
    evaluate()
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    torch.cuda.empty_cache()
