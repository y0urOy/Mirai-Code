from dataset.imagenet import build_imagenet_code

import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder # 关键 import
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import random
from dataset.augmentation import center_crop_arr
import torchvision.transforms.functional as F

def get_indexed_crop(pil_image, target_size, index):
    """
    根据索引 (0-9) 从一张 PIL 图像中获取唯一的裁剪图，模拟 TenCrop 的行为。
    
    索引对应关系:
    - 0-4: tencrop 的前5个 (不翻转)
        - 0: 左上 (Top-Left)
        - 1: 右上 (Top-Right)
        - 2: 左下 (Bottom-Left)
        - 3: 右下 (Bottom-Right)
        - 4: 中心 (Center)
    - 5-9: 对应0-4的水平翻转版本
    
    Args:
        pil_image (PIL.Image): 输入的 PIL 图像。
        target_size (int): 裁剪的目标尺寸 (正方形)。
        index (int): 0到9之间的索引。

    Returns:
        PIL.Image: 单一的裁剪后的 PIL 图像。
    """
    if index >= 5:
        pil_image = F.hflip(pil_image)
        crop_index = index - 5
    else:
        crop_index = index

    w, h = pil_image.size
    
    # FiveCrop 的顺序是：TL, TR, BL, BR, Center
    if crop_index == 0:  # Top-Left
        return F.crop(pil_image, 0, 0, target_size, target_size)
    elif crop_index == 1:  # Top-Right
        return F.crop(pil_image, 0, w - target_size, target_size, target_size)
    elif crop_index == 2:  # Bottom-Left
        return F.crop(pil_image, h - target_size, 0, target_size, target_size)
    elif crop_index == 3:  # Bottom-Right
        return F.crop(pil_image, h - target_size, w - target_size, target_size, target_size)
    elif crop_index == 4:  # Center
        return F.center_crop(pil_image, (target_size, target_size))
    else:
        raise ValueError(f"Crop index must be 0-4, but got {crop_index}")

def center_crop_dhariwal(pil_image, image_size):
    """
    这是 REPA 官方预处理脚本中 center_crop_imagenet 方法的直接适配版本。
    它接收一个 PIL 图像，并返回一个经过高质量中心裁剪和缩放的 PIL 图像。
   
    """
    # 渐进式下采样以抗锯齿
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=Image.Resampling.BOX)

    # 高质量缩放
    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)

    # 中心裁剪
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    cropped_arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    return Image.fromarray(cropped_arr)

def build_dataset(args, **kwargs):
    if args.dataset == 'imagenet_json':
        crop_range= 1.1 
        image_size = 256
        crop_size = int(image_size * crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return JsonDataset(
            json_path=args.json_path,
            image_size=args.image_size,
            # transform=transform  # <-- 使用统一的 transform
        )
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')

class ImageFolderWithPaths(ImageFolder):
    """
    一个自定义的 ImageFolder，它的 __getitem__ 会额外返回图像的路径。
    这是确保在 DataLoader 中能够追踪到原始文件来源的关键。
    """
    def __getitem__(self, index):
        # 1. 调用父类来获取原始的图像张量和标签
        original_tuple = super().__getitem__(index)
        # 2. 获取该索引对应的原始路径
        path = self.samples[index][0]
        # 3. 将路径添加到返回值元组中
        #    现在返回值是 (图像张量, 标签, 路径)
        return (*original_tuple, path)
    
class JsonDataset(Dataset):
    def __init__(self, json_path, image_size=256): # 直接传入 image_size
        super().__init__()
        self.image_size = image_size
        
        # 预处理中可能仍然需要的第一步 (放大图像以便后续裁剪)
        self.crop_range = 1.1
        self.initial_crop_size = int(self.image_size * self.crop_range)

        # 不再需要重量级的 self.transform
        self.to_tensor = transforms.ToTensor()
        
        print(f"从清单文件加载 REPA 数据集: {json_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"清单文件不存在: {json_path}")
            
        with open(json_path, 'r') as f:
            self.manifest = json.load(f)
            
        print(f"✅ 成功加载 {len(self.manifest)} 个样本的元数据。")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        sample_info = self.manifest[idx]
        
        # --- 核心逻辑修改 ---
        
        # 1. 加载 VQGAN codes
        code_path = sample_info["code_path"]
        all_codes = np.load(code_path) # shape: (1, 10, seq_len)
        
        # 2. ✅ 步骤 1: 首先决定随机使用哪个增强版本
        num_augmentations = all_codes.shape[1] # 应该是 10
        aug_index = random.randint(0, num_augmentations - 1)
        
        # 3. ✅ 步骤 2: 根据索引获取对应的 VQGAN code
        selected_codes_np = all_codes[0, aug_index, :]
        vqgan_codes = torch.from_numpy(selected_codes_np)
        
        # 4. ✅ 步骤 3: 只对图像执行一次对应的处理
        image_path = sample_info["image_path"]
        raw_image_pil = Image.open(image_path).convert('RGB')
        
        # 4.1 执行初始的放大裁剪
        processed_pil = center_crop_arr(raw_image_pil, self.initial_crop_size)
        
        # # 4.2 ✅ 根据索引，只生成一张裁剪图
        final_crop_pil = get_indexed_crop(processed_pil, self.image_size, aug_index)
        
        # 4.3 转换为 Tensor
        raw_image = self.to_tensor(final_crop_pil)

        # 5. 加载标签 (逻辑不变)
        label_path = sample_info["label_path"]
        label = np.load(label_path)[0]
        
        return raw_image, vqgan_codes, torch.tensor(label, dtype=torch.long)