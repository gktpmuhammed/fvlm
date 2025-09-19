import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from monai import transforms
from monai.data.utils import dense_patch_slices
from typing import Any, Callable, List, Sequence, Tuple, Union

from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.common.dist_utils import get_rank, init_distributed_mode


def masks_to_boxes_3d(masks):
    """Compute the bounding boxes around the provided 3D masks

    The masks should be in format [N, D, H, W] where N is the number of masks, (D, H, W) are the spatial dimensions.

    Returns a [N, 6] tensor, with the boxes in min_x, min_y, min_z, max_x, max_y, max_z format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    d, h, w = masks.shape[-3:]

    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)

    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)

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

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
        """
        Compute scan interval according to the image size, roi size and overlap.
        Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
        use 1 instead to make sure sliding window works.

        """
        if len(image_size) != num_spatial_dims:
            raise ValueError("image coord different from spatial dims.")
        if len(roi_size) != num_spatial_dims:
            raise ValueError("roi coord different from spatial dims.")

        scan_interval = []
        for i in range(num_spatial_dims):
            if roi_size[i] == image_size[i]:
                scan_interval.append(int(roi_size[i]))
            else:
                interval = int(roi_size[i] * (1 - overlap))
                scan_interval.append(interval if interval > 0 else 1)
        return tuple(scan_interval)

def center_crop(image, mask, crop_size):
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()
    
    crop_d, crop_h, crop_w = max(crop_size[0], z_max - z_min), max(crop_size[1], y_max - y_min), max(crop_size[2], x_max - x_min)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2
    
    d, h, w = image.shape[-3:]

    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)
    
    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)
    
    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)
    
    return image[..., z_start:z_end, y_start:y_end, x_start:x_end], mask[..., z_start:z_end, y_start:y_end, x_start:x_end]

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
    
    fused_mask = np.zeros_like(label)
    for original_id, organ_name in class_map.items():
        if organ_name in merged_organ_id:
            merged_id = merged_organ_id[organ_name]
            fused_mask[label == original_id] = merged_id + 1
    return fused_mask

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

class DataFolder(Dataset):
    def __init__(self, vis_root='data/processed_valid_images'):
        super().__init__()

        img_paths = []
        for root, _, files in os.walk(vis_root):
            for file in files:
                if file.endswith('.nii.gz'):
                    img_paths.append(os.path.join(root, file))
        self.img_paths = img_paths

        self.organs = [
            'lung', 'heart', 'esophagus', 'aorta'
        ]

        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
            transforms.Lambdad(keys=["label"], func=merge_labels),
            SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            Original_ROI_Crop_d(keys=["image", "label"]),
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
            ToRegularTensor(keys=["image", "label"])
        ])
        
        self.pathologies = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']

        self.test_items = [
            ['lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'],
            ['lung', 'Atelectasis', 'Not Atelectatic.', 'Atelectatic.'], 
            ['lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'],
            ['lung', 'Lung opacity', 'Not Opacity.', 'Opacity.'],
            ['lung', 'Pulmonary fibrotic sequela', 'Not Pulmonary fibrotic.', 'Pulmonary fibrotic.'],
            ['lung', 'Pleural effusion', 'Not Pleural effusion.', 'Pleural effusion.'],
            ['lung', 'Mosaic attenuation pattern', 'Not Mosaic attenuation pattern.', 'Mosaic attenuation pattern.'],
            ['lung', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'],
            ['lung', 'Consolidation', 'Not Consolidation.', 'Consolidation.'],
            ['lung', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'],
            ['lung', 'Interlobular septal thickening', 'Not Interlobular septal thickening.', 'Interlobular septal thickening.'],
            ['heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'],
            ['heart', 'Pericardial effusion', 'Not Pericardial effusion.', 'Pericardial effusion.'],
            ['heart', 'Coronary artery wall calcification', 'Not Coronary artery wall calcification.', 'Coronary artery wall calcification.'],
            ['esophagus', 'Hiatal hernia', 'Not Hiatal hernia.', 'Hiatal hernia.'],
            ['aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'],
        ]
        
        self.test_items = [tuple(item) for item in self.test_items]

        self.test_organs = list(set([item[0] for item in self.test_items]))

        if dist.is_initialized():
            self.img_paths = self.img_paths[dist.get_rank()::dist.get_world_size()]

    @staticmethod
    def get_patient_id(image_path):
        img_name = image_path.split('/')[-1]
        patient_id = img_name.split('_')[0]
        return patient_id
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = image_path.replace('images', 'masks')
        
        file_name = image_path.split('/')[-1]
        input_path = {'image': image_path, 'label': mask_path}

        data = self.transform(input_path)
        
        test_organ_names = self.organs
        test_items = [test_item for test_item in self.test_items if test_item[0] in test_organ_names]
        
        meta_info = {
            'file_name': file_name,
            'img_path': image_path,
            'test_organ_names': test_organ_names
        }
        
        return data['image'].as_tensor(), data['label'].as_tensor(), test_items, meta_info

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--csv_file', type=str, help='The path to the CSV file for processing.')
    parser.add_argument('--vis_root', type=str, default='data/processed_valid_images', help='The path to the visual root directory.')
    parser.add_argument('--ckpt_path', type=str, default='/home/muhammedg/fvlm/checkpoints/model.pth', help='The path to the checkpoint file.')

    parser.add_argument("--cfg-path", required=False, default='lavis/projects/blip/train/pretrain_ct.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

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
        batch_size=1,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    pad_func = transforms.DivisiblePadd(
                    keys=["image", "label"], 
                    k=(16, 16, 32),
                    mode='constant', 
                    constant_values=0,
                    method="end"
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

    # Set global precision for printing tensors
    torch.set_printoptions(precision=2)
    
    sw_batch_size = 4

    overlap = 0.25
    roi_size = (112, 288, 352)

    results = []
    
    text_feat_dict = model.prepare_text_feat(datafolder.test_items)

    organ_feat_dict = {}

    save_path = '_'.join(ckpt_path.replace('.pth', '').split('/')[1:])

    for i, (image, mask, test_items, meta_info) in enumerate(tqdm(dataloader, desc='Infer')):
        fid = meta_info['file_name']
        organ_feat_dict[fid] = {}

        image = image[None].cuda()
        mask = mask[None].cuda()

        test_organs = meta_info['test_organ_names']
        
        whole_organ_sizes = dict(zip(test_organs, [torch.eq(mask, datafolder.organs.index(test_organ) + 1).sum().item() for test_organ in test_organs]))    

        test_organs = [test_organ for test_organ in test_organs if whole_organ_sizes[test_organ] > 0]
        test_items = [test_item for test_item in test_items if test_item[0] in test_organs]

        image_size = list(image.shape[2:])
        num_spatial_dims = len(image.shape) - 2

        scan_interval = _get_scan_interval(
            image_size, roi_size, num_spatial_dims, overlap
        )
        slices = dense_patch_slices(image_size, roi_size, scan_interval)
        num_win = len(slices)

        organ_logits = dict(zip(test_items, [[] for _ in test_items]))
        for k, v in organ_logits.items():
            if not len(v):
                organ_name = k[0]
                organ_id = datafolder.organs.index(organ_name)

                window_patch, window_mask = center_crop(
                    image,
                    torch.eq(mask, organ_id + 1),
                    crop_size=roi_size
                )
                window_mask = window_mask.float()
                window_mask[window_mask == 1] = organ_id + 1

                pad_data = pad_func({'image': window_patch[0], 'label': window_mask[0]})
                window_patch, window_mask = pad_data['image'], pad_data['label']

                # print('EXTRA', organ_name, window_patch.size())

                organ_logits = model.forward_test_win(
                    window_patch[None], 
                    window_mask[None],
                    organ_logits,
                    test_organs,
                    text_feat_dict,
                    organ_feat_dict[fid],
                    whole_organ_sizes,
                    skip_organ=organ_id
                )
            
        res = [meta_info['file_name']] + [''] * len(datafolder.test_items)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        for item, probs in organ_logits.items():
            res[datafolder.test_items.index(item) + 1] = np.concatenate(probs).mean(0)[1]
        results.append(res)
    
    if dist.is_initialized():
        results = np.concatenate(all_gather(results), axis=0)
        organ_feat_dict = all_gather(organ_feat_dict)
    else:
        organ_feat_dict = [organ_feat_dict]
    
    if rank == 0:
        os.makedirs('rate_res', exist_ok=True)
        pd.DataFrame(
            results,
            columns=['file_name'] + ['_'.join(k) for k in datafolder.test_items]
        ).to_csv(f'rate_res/{save_path}.csv', index=False, encoding='utf-8')
        
        print('Save csv file successfully!')

if __name__ == '__main__':
    evaluate()
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    torch.cuda.empty_cache()
