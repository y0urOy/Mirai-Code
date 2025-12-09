# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy

# 关键修复点 2：导入 allow_in_graph
from torch.compiler import allow_in_graph

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

import os
import sys
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import inspect
import argparse
import wandb

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt_repa_two_head import GPT_models # 确保这里导入的是您 two_head 版本的模型
from torch.nn import functional as F # <-- 确保导入
import timm # <-- 新增 import
import math # <-- 确保在文件顶部导入 math 库


@allow_in_graph
def calculate_repa_loss_multi_head_corrected(zs_teacher_list, zs_tilde_list):
    """
    计算多头预测的 REPA 损失 (最终修正版)。
    这个版本直接对整个序列进行切片，避免了 CLS 和 SEQ 分离带来的对齐错误。
    """
    head_losses = []
    
    # 教师特征只有一个来源
    z_teacher = zs_teacher_list[0]
      
    
    # 遍历学生模型的每一个预测头
    for i, z_student in enumerate(zs_tilde_list):
        # --- 新增逻辑: 根据头索引 i 确定预测偏移量 offset ---
        if i == 0:
            offset = 0
        elif i == 1:
            offset = 1
        elif i == 2:
            offset = 16
        elif i == 3:
            offset = 17
        elif i == 4:
            offset = 2
        elif i == 5:
            offset = 18
        elif i == 6:
            offset = 34
        elif i == 7:
            offset = 33
        elif i == 8:
            offset = 32
        else:
            # 为其他可能的头提供一个默认行为，例如 offset = i
            # 或者您可以根据需要抛出错误或设置为固定值
            offset = i
            
        # 1. 分离学生的 CLS 
        z_student = z_student[:, 1:, :]
        
        # 2. 对 Patch 部分进行时序对齐 (使用自定义的 offset)
        # 确定教师的目标 Patch 序列
        if z_teacher.size(1) <= offset:
            # 如果教师 patch 序列不够长，无法移位，则此头损失为0
            head_losses.append(torch.tensor(0.0, device=z_teacher.device, dtype=z_teacher.dtype))
            continue
            
        target_teacher_patch = z_teacher[:, offset:, :]
        
        # 截取学生 patch 序列的前面部分，以匹配长度
        common_patch_len = target_teacher_patch.size(1)
        student_prediction_patch = z_student[:, :common_patch_len, :]
        
        
        # # 确保拼接后的长度一致
        # assert target_teacher_patch.size(1) == student_prediction_patch.size(1)
        if student_prediction_patch.shape[1] > target_teacher_patch.shape[1]:
            student_prediction_patch = student_prediction_patch[:, :target_teacher_patch.shape[1]]
        elif target_teacher_patch.shape[1] > student_prediction_patch.shape[1]:
            target_teacher_patch = target_teacher_patch[:, :student_prediction_patch.shape[1]]
            
        # 4. 对拼接好的、完全对齐的张量计算总损失
        zt_norm = F.normalize(target_teacher_patch, p=2, dim=-1)
        zs_norm = F.normalize(student_prediction_patch, p=2, dim=-1)
        
        loss = (-(zt_norm * zs_norm).sum(dim=-1)).mean()
        head_losses.append(loss)
            
    return head_losses

def get_piecewise_coeff_multiplier(current_epoch: int) -> float:
    """
    分段常数系数：
      0-99  epoch: 2.0
      100-199 epoch: 0.5
      >=200  epoch: 0.25
    """
    if current_epoch < 80:
        return 1.0
    else:
        return 0.5

def preprocess_raw_image(x):
    resolution = x.shape[-1]
    # x = x / 255.
    # x = x.to(torch.float32) / 255.0
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if 'dinov2' in encoder_type:
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures

#################################################################################
#                           Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}    
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

#################################################################################
#                                 Training Loop                                 #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.report_to_wandb:
            wandb.init(
                project=args.wandb_project,
                name=os.path.basename(experiment_dir),
                config=args
            )
            logger.info("Weights & Biases initialized.")
    else:
        logger = create_logger(None)

    logger.info(f"Training args: {args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    #教师模型
    logger.info(f"Loading teacher encoder: {args.enc_type}")
    encoders, encoder_types, architectures = load_encoders(args.enc_type, device, args.resolution)
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]


    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
        

    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        encoder_depth=args.student_depth,
        z_dims = z_dims,
        num_repa_heads=args.num_repa_heads
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

        # 确保教师编码器不参与训练
    for encoder in encoders:
        encoder.eval()
        requires_grad(encoder, False)

    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory_device="cuda",
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images.")

    # --- 新增：为动态计算做准备 ---
    # `len(loader)` 能准确计算出每个rank上，一个epoch有多少个step
    steps_per_epoch = len(loader)

    # 1. 新增：根据 warmup_epochs 计算出总的 warmup_steps
    warmup_steps_total = args.warmup_epochs * steps_per_epoch

    # 2. 决定衰减的总epoch数 (逻辑不变)
    decay_epochs = args.proj_coeff_decay_epoch if args.proj_coeff_decay_epoch > 0 else 0

    # 3. 辅助函数需要的 warmup 进度现在可以直接使用参数 (逻辑简化)
    warmup_in_epochs = float(args.warmup_epochs)
    
    if rank == 0:
        logger.info(f"Warmup is set to {args.warmup_epochs} epochs, which is equivalent to {warmup_steps_total} steps.")
        if decay_epochs > 0:
            logger.info(f"REPA coeffs will decay over {decay_epochs} epochs, starting AFTER warmup.")

    # --------------------------------

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    
    train_steps = 0
    start_epoch = 0

    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.ema:
            ema.load_state_dict(checkpoint["ema"], strict=False)
          
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
             scaler.load_state_dict(checkpoint["scaler"])
        
        train_steps = checkpoint["steps"]
        start_epoch = checkpoint["epoch"] + 1 # Start from the next epoch

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
            logger.info(f"Restored RNG state from checkpoint.")

        del checkpoint
        logger.info(f"Resuming training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, start_epoch={start_epoch}")
    elif args.ema:
        update_ema(ema, model, decay=0)

    if not args.no_compile:
        logger.info("Compiling the model... (may take several minutes)")
        model = torch.compile(model)
    
    # DDP包装：find_unused_parameters=True 是保证多头模型正确运行的安全设置。
    # PyTorch可能会发出一个性能警告，但在确定所有头始终有梯度之前，建议保留此设置并忽略该警告。
    # model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    model = DDP(model, device_ids=[args.gpu])

    model.train()
    if args.ema:
        ema.eval()
    
    log_steps = 0
    running_loss = 0
    running_loss_repa_heads = [0.0] * args.num_repa_heads
    running_loss_primary = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch_idx, (raw_image, x, y) in enumerate(loader):
            raw_image = raw_image.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                _, zs_tilde, primary_loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)

            # 无论是否在 warmup 阶段，都计算 REPA loss，但在 warmup 阶段让其结果为0或不生效
            if train_steps >= warmup_steps_total:
                processed_image = preprocess_raw_image(raw_image)
                teacher_features = encoders[0].forward_features(processed_image)
                if 'dinov2' in encoder_types[0]:
                    teacher_features = teacher_features['x_norm_patchtokens']


                zs_teacher_list = [teacher_features]
                
                repa_loss_list = calculate_repa_loss_multi_head_corrected(zs_teacher_list, zs_tilde)
                # --- 新增：计算当前的余弦退火系数 ---
                current_epoch_progress = epoch + (batch_idx / steps_per_epoch)

                # --- 2. 修改这里的函数调用 ---
                # 原来的 warmup_epoch_progress 来自于一个除法计算
                # decay_multiplier = get_cosine_decay_multiplier_by_epoch(
                #     current_epoch_progress=current_epoch_progress,
                #     warmup_epoch_progress=warmup_in_epochs, # warmup_in_epochs 现在直接等于 args.warmup_epochs
                #     total_decay_epochs=decay_epochs
                # )
                # current_proj_coeffs = [c * decay_multiplier for c in args.proj_coeffs]

                coeff_mult = get_piecewise_coeff_multiplier(epoch)
                current_proj_coeffs = [c * coeff_mult for c in args.proj_coeffs]
                # ------------------------------------

                total_repa_loss = 0
                for i in range(args.num_repa_heads):
                    # 使用动态计算的系数
                    total_repa_loss += repa_loss_list[i] * current_proj_coeffs[i]

                loss = primary_loss + total_repa_loss
            else:
                # 在 warmup 阶段，创建一个假的 repa_loss 来确保计算图连接
                # 这个 loss 的值是 0，但它连接了 zs_tilde，从而确保了梯度的反向传播
                fake_repa_loss = 0.0
                for z in zs_tilde:
                    fake_repa_loss += (z * 0).sum() # 乘以0，值是0，但计算图是连接的
                    
                loss = primary_loss + fake_repa_loss
                # 也可以保留 repa_loss_list 用于日志记录
                repa_loss_list = [torch.tensor(0.0, device=device) for _ in range(args.num_repa_heads)]

            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)


            running_loss += loss.item()
            for i in range(args.num_repa_heads):
                running_loss_repa_heads[i] += repa_loss_list[i].item()
            running_loss_primary += primary_loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # 同步和记录日志
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_primary = torch.tensor(running_loss_primary / log_steps, device=device)
                avg_loss_repa_heads = [torch.tensor(val / log_steps, device=device) for val in running_loss_repa_heads]

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_primary, op=dist.ReduceOp.SUM)
                for i in range(args.num_repa_heads):
                    dist.all_reduce(avg_loss_repa_heads[i], op=dist.ReduceOp.SUM)
                
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_primary = avg_loss_primary.item() / dist.get_world_size()
                avg_loss_repa_heads = [val.item() / dist.get_world_size() for val in avg_loss_repa_heads]

                log_msg = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Primary Loss: {avg_loss_primary:.4f}, "
                wandb_log = {
                    "total_loss": avg_loss,
                    "primary_loss": avg_loss_primary,
                    "steps_per_sec": steps_per_sec,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }   
                
                wandb_log["repa_coeff_multiplier"] = get_piecewise_coeff_multiplier(epoch)

                for i in range(args.num_repa_heads):
                    log_msg += f"Repa Loss_{i+1}: {avg_loss_repa_heads[i]:.4f}, "
                    wandb_log[f"repa_loss_{i+1}"] = avg_loss_repa_heads[i]
                log_msg += f"Train Steps/Sec: {steps_per_sec:.2f}"
                logger.info(log_msg)

                if rank == 0 and args.report_to_wandb:
                    wandb.log(wandb_log, step=train_steps)
                
                running_loss = 0
                running_loss_repa_heads = [0.0] * args.num_repa_heads
                running_loss_primary = 0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    model_weight = model.module._orig_mod.state_dict() if not args.no_compile else model.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "steps": train_steps,
                        "epoch": epoch,
                        "rng_state": torch.get_rng_state(),
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    if rank == 0 and args.report_to_wandb:
        wandb.finish()
    logger.info("Done!")
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... 您原有的所有 parser.add_argument(...) 代码都保持不变 ...
    # 我只列出需要确保存在的几个关键参数
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True)
    parser.add_argument("--no-local-save", action='store_true')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--student-depth", type=int, default=8)
    parser.add_argument("--teacher-depth", type=int, default=8)
    parser.add_argument("--dataset", type=str, required=True) # 确保dataset参数存在
    parser.add_argument("--raw-image-path", type=str, required=True)
    parser.add_argument("--proj-coeffs", type=float, nargs='+', default=[0.25, 0.25])
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, default="LlamaGen-REPA_3_head")
    parser.add_argument("--report-to-wandb", action='store_true')
    # parser.add_argument("--use-prev-iter-ema", action='store_true')
    # parser.add_argument("--warmup-steps", type=int, default=75000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b', help="Type of teacher encoder for REPA.")
    parser.add_argument("--num-repa-heads", type=int, default=2) # 确保有这个参数
    parser.add_argument("--proj-coeff-decay-epoch", type=int, default=0,
                        help="Total steps for cosine decay of proj_coeffs. "
                             "Decay starts after warmup. If <= 0, no decay is applied.")

    args = parser.parse_args()
    main(args)