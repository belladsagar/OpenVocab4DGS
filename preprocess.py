import os
import random
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to(self.device)
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        
        # Calculate embeddings on specific device
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(self.device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    # Ensure operations happen on the device of the masks
    device = masks.device
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=device)
    
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            # Fix for potential division by zero if mask is empty, though unlikely with SAM
            if masks_area[i] == 0 or masks_area[j] == 0:
                continue
                
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # Safety checks for empty results
    if keep_conf.sum() == 0 and len(scores) > 0:
        index = scores.topk(min(3, len(scores))).indices 
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0 and len(scores) > 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0 and len(scores) > 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_l[index, 0] = True
        
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(device, *args, **kwargs):
    masks_new = ()
    for masks_lvl in (args):
        if len(masks_lvl) == 0:
            masks_new += ([],)
            continue
            
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0)).to(device)
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0)).to(device)
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0)).to(device)

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image, mask_generator, device):
    # Image comes in as (1, C, H, W) tensor
    image_np = image[0].permute(1,2,0).numpy()
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # SAM inference
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image_cv)
    
    # Postprocess / NMS
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(device, masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image, original_shape, device):
        seg_img_list = []
        seg_map = -np.ones((original_shape[0], original_shape[1]), dtype=np.int32)
        
        if len(masks) == 0:
             return torch.tensor([]).to(device), seg_map

        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
            
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to(device)

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    h, w = image_cv.shape[:2]
    
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image_cv, (h,w), device)
    
    seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image_cv, (h,w), device) if len(masks_s) > 0 else (torch.tensor([]).to(device), -np.ones((h,w), dtype=np.int32))
    seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image_cv, (h,w), device) if len(masks_m) > 0 else (torch.tensor([]).to(device), -np.ones((h,w), dtype=np.int32))
    seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image_cv, (h,w), device) if len(masks_l) > 0 else (torch.tensor([]).to(device), -np.ones((h,w), dtype=np.int32))
    
    return seg_images, seg_maps

def _embed_clip_sam_tiles(image, mask_generator, clip_model, device):
    # Refactored to take models as args
    seg_images, seg_map = sam_encoder(image, mask_generator, device)

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode]
        if tiles.shape[0] == 0:
             clip_embeds[mode] = torch.tensor([]).half()
             continue
             
        tiles = tiles.to(device)
        with torch.no_grad():
            clip_embed = clip_model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

def process_subset(rank, world_size, dataset_path, sam_ckpt_path, resolution_scale, subset_files):
    """
    Worker function to process a subset of images on a specific GPU
    """
    device = f"cuda:{rank}"
    print(f"[GPU {rank}] Starting... Processing {len(subset_files)} images.")
    
    # 1. Setup Directories
    img_folder = os.path.join(dataset_path, 'images')
    save_folder = os.path.join(dataset_path, 'language_features')
    
    # 2. Initialize Models on this Rank's Device
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig, device)
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path)
    sam.to(device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    # 3. Process Loop
    # Only show progress bar on rank 0 to reduce clutter, or use position argument
    iterator = tqdm(subset_files, desc=f"GPU {rank}", position=rank)

    for filename in iterator:
        image_path = os.path.join(img_folder, filename)
        
        # Lazy Load Image (Don't load all to RAM at start)
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            print(f"[GPU {rank}] Warning: Could not load {image_path}")
            continue

        # Resize
        orig_w, orig_h = image_cv.shape[1], image_cv.shape[0]
        if resolution_scale == -1:
            scale = 1.0
        else:
            scale = resolution_scale
            
        resolution = (int(orig_w / scale), int(orig_h / scale))
        image_cv = cv2.resize(image_cv, resolution)
        
        # Convert to torch
        image_tensor = torch.from_numpy(image_cv).permute(2, 0, 1)[None, ...]
        
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(image_tensor, mask_generator, clip_model, device)
        except Exception as e:
            print(f"[GPU {rank}] Error processing {filename}: {e}")
            continue

        # Aggregate Embeddings
        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        
        if total_length == 0:
            # Handle case where no masks were found
            continue

        img_embed_cat = torch.cat([v for k, v in img_embed.items()], dim=0)

        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
            
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
            
        current_seg_map = torch.stack(seg_map_tensor, dim=0)
        save_path_base = os.path.join(save_folder, filename.split('.')[0])
        
        # Save
        np.save(save_path_base + '_s.npy', current_seg_map.numpy())
        np.save(save_path_base + '_f.npy', img_embed_cat.numpy())

    print(f"[GPU {rank}] Finished.")

def main_worker(rank, world_size, args, data_list):
    # Seed everything per process to ensure deterministic behavior if needed
    # but vary slightly per rank if random transforms were used (not here, but good practice)
    seed = 42 + rank
    seed_everything(seed)
    
    # Split data logic: 
    # GPU 0 gets indices 0, 4, 8...
    # GPU 1 gets indices 1, 5, 9...
    my_files = data_list[rank::world_size]
    
    process_subset(rank, world_size, args.dataset_path, args.sam_ckpt_path, args.resolution, my_files)

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.float32)

    # Prepare file list in main process
    img_folder = os.path.join(args.dataset_path, 'images')
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"Image folder not found: {img_folder}")
        
    data_list = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    data_list.sort()
    
    save_folder = os.path.join(args.dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)

    world_size = args.gpus
    print(f"Spawning {world_size} processes for {len(data_list)} images...")
    
    # Spawn processes
    mp.spawn(
        main_worker,
        args=(world_size, args, data_list),
        nprocs=world_size,
        join=True
    )