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

# Remove organ-aware segmentation code - not needed for simplified report generation
class SpacingNormalization:
    """Custom transform to normalize spacing to reference spacing"""
    def __init__(self, ref_spacing=(1.0, 1.0, 3.0), debug=False):
        self.ref_spacing = ref_spacing
        self.debug = debug
    
    def __call__(self, data):
        affine = data["image_meta_dict"]["affine"]
        spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
        
        _, h, w, d = data["image"].shape
        scale = [spacing[i] / self.ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]
        
        if target_size != [h, w, d]:
            resize_transform = transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear")
            return resize_transform(data)
        else:
            return data

# Removed merge_labels function - not needed for simplified report generation

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

        # Simplified transform - only process images (no segmentation masks needed)
        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
            SpacingNormalization(ref_spacing=(1.0, 1.0, 3.0)),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=(112, 256, 352),
                mode="constant",
                constant_values=0
            ),
            transforms.CenterSpatialCropd(
                keys=["image"],
                roi_size=(112, 256, 352)
            ),
        ])
        
        if dist.is_initialized():
            self.img_paths = self.img_paths[dist.get_rank()::dist.get_world_size()]
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        file_name = image_path.split('/')[-1]
        
        # Only load image - no segmentation mask needed for report generation
        data = self.transform({"image": image_path})
        
        return {
            "image": data['image'],
            "file_name": file_name
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Simplified Report Generation Evaluation")
    parser.add_argument('--vis_root', type=str, default='data/valid/images', help='The path to the visual root directory.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='The path to the trained checkpoint file.')
    parser.add_argument("--cfg-path", required=False, default='lavis/projects/blip/train/finetune_report_generation.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # Generation parameters - optimized for medical reports
    parser.add_argument("--use_nucleus_sampling", action='store_true', default=True, help="Use nucleus sampling")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams (1 for nucleus sampling)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--min_length", type=int, default=25, help="Minimum generation length")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")

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
    
    for i, samples in enumerate(tqdm(dataloader, desc='Generating Medical Reports')):
        samples["image"] = samples["image"].cuda()
        
        # Generate report using simplified approach: Image → Vision Encoder → Text Decoder
        generated_text = model.generate(
            samples,
            use_nucleus_sampling=args.use_nucleus_sampling,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )

        results.append({
            "file_name": samples["file_name"], 
            "report": generated_text[0],
            "generation_params": {
                "use_nucleus_sampling": args.use_nucleus_sampling,
                "num_beams": args.num_beams,
                "max_length": args.max_length,
                "min_length": args.min_length,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty
            }
        })
    
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
