import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import json
from tqdm import tqdm
import datetime

import os
import sys


try:
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    pass

from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    if not args.debug:
        timeout = datetime.timedelta(hours=2)
        init_distributed_mode(args, timeout=timeout)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device, rank, args.world_size = 'cuda', 0, 1
    
    codes_dir = os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes')
    labels_dir = os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels')
    final_json_path = os.path.join(args.code_path, f"{args.dataset}_{args.image_size}_manifest.json")
    
    if args.debug or rank == 0:
        os.makedirs(codes_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    vq_model = VQ_models[args.vq_model](codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim).to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), 
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, seed=args.global_seed) if not args.debug else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    tmp_manifest_path = f"/tmp/manifest_rank_{rank}.jsonl"
    with open(tmp_manifest_path, 'w') as f_manifest:
        total = 0
        pbar = tqdm(loader) if rank == 0 else loader
        
        for x, y, image_paths in pbar:
            image_path = image_paths[0]

            x = x.to(device)
            if args.ten_crop:
                x_all = x.flatten(0, 1); num_aug = 10
            else:
                x_flip = torch.flip(x, dims=[-1]); x_all = torch.cat([x, x_flip]); num_aug = 2
            
            with torch.no_grad():
                _, _, [_, _, indices] = vq_model.encode(x_all)
            codes = indices.reshape(x.shape[0], num_aug, -1)
            
            train_steps = rank + total
            
            code_np = codes.detach().cpu().numpy()
            code_path = os.path.join(codes_dir, f"{train_steps}.npy")
            np.save(code_path, code_np)

            label_np = y.numpy()
            label_path = os.path.join(labels_dir, f"{train_steps}.npy")
            np.save(label_path, label_np)
            
            sample_entry = {
                "image_path": os.path.abspath(image_path),
                "code_path": os.path.abspath(code_path),
                "label_path": os.path.abspath(label_path),
                "id": train_steps
            }
            f_manifest.write(json.dumps(sample_entry) + '\n')

            if not args.debug:
                total += dist.get_world_size()
            else:
                total += 1
            
            if rank == 0:
                pbar.set_description(f"Processed {total} images")
    
    print(f"Rank {rank} finished processing and saved temporary manifest to {tmp_manifest_path}")

    if not args.debug:
        dist.barrier()
        if rank == 0:
            print("Rank 0 is merging all temporary manifests...")
            all_entries = []
            for i in range(dist.get_world_size()):
                tmp_path = f"/tmp/manifest_rank_{i}.jsonl"
                with open(tmp_path, 'r') as f_temp:
                    for line in f_temp:
                        all_entries.append(json.loads(line))
                os.remove(tmp_path)
            
            all_entries.sort(key=lambda x: x['id'])
            
            with open(final_json_path, 'w') as f_final:
                json.dump(all_entries, f_final, indent=4)
            print(f"✅ Final manifest saved to: {final_json_path} (Total: {len(all_entries)} records)")

    if not args.debug:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True, help="保存 codes/labels 和 json 的最终根目录")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true', help="whether using random crop")
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)