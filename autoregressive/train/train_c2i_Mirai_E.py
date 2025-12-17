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

from torch.compiler import allow_in_graph

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

import os
import sys
# current_file_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import time
import inspect
import argparse
import wandb

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.Mirai import GPT_models 
from torch.nn import functional as F 
import timm 
import math 

@allow_in_graph
def calculate_foresight_loss(zs_teacher_list, zs_tilde_list):
    head_losses = []
    
    z_teacher_full = zs_teacher_list[0]
    t_cls = z_teacher_full[:, 0:1, :] 
    t_patch = z_teacher_full[:, 1:, :]
    
    for i, z_student_full in enumerate(zs_tilde_list):
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
            offset = i
            
        s_cls = z_student_full[:, 0:1, :]
        s_patch = z_student_full[:, 1:, :]

        if t_patch.size(1) <= offset:
            head_losses.append(torch.tensor(0.0, device=t_patch.device, dtype=t_patch.dtype))
            continue
            
        target_teacher_patch = t_patch[:, offset:, :]
        
        common_patch_len = target_teacher_patch.size(1)
        student_prediction_patch = s_patch[:, :common_patch_len, :]
        
        final_target_teacher = torch.cat([t_cls, target_teacher_patch], dim=1)
        final_student_prediction = torch.cat([s_cls, student_prediction_patch], dim=1)
        
        assert final_target_teacher.size(1) == final_student_prediction.size(1)
        
        zt_norm = F.normalize(final_target_teacher, p=2, dim=-1)
        zs_norm = F.normalize(final_student_prediction, p=2, dim=-1)
        
        loss = (-(zt_norm * zs_norm).sum(dim=-1)).mean()
        head_losses.append(loss)
            
    return head_losses

def get_piecewise_coeff_multiplier(current_epoch: int) -> float:

    if current_epoch < 80:
        return 1.0
    else:
        return 0.5

   
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

    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
        
    ema_teacher = None
    if args.use_prev_iter_ema:
        logger.info(f"Loading teacher ema model...")
        latent_size = args.image_size // args.downsample_size
        ema_teacher = GPT_models[args.gpt_model](
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=args.drop_path_rate,
            token_dropout_p=args.token_dropout_p,
            encoder_depth=args.teacher_depth,
            z_dims=[768],
            num_heads=args.num_heads
        ).to(device)
        ema_teacher.eval()
        requires_grad(ema_teacher, False)

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
        z_dims=[768],
        num_heads=args.num_heads
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

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

    steps_per_epoch = len(loader)

    warmup_steps_total = args.warmup_epochs * steps_per_epoch

    decay_epochs = args.proj_coeff_decay_epoch if args.proj_coeff_decay_epoch > 0 else 0
    
    if rank == 0:
        logger.info(f"Warmup is set to {args.warmup_epochs} epochs, which is equivalent to {warmup_steps_total} steps.")
        
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
            if args.use_prev_iter_ema:
                logger.info("Syncing ema_teacher from resumed EMA state...")
                ema_teacher.load_state_dict(ema.state_dict())
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

    model = DDP(model, device_ids=[args.gpu])

    model.train()
    if args.ema:
        ema.eval()
    
    if args.use_prev_iter_ema and not args.gpt_ckpt:
        if rank == 0:
            logger.info("Syncing initial teacher model with student model...")
        unwrapped_model = model.module._orig_mod if not args.no_compile else model.module
        ema_teacher.load_state_dict(unwrapped_model.state_dict())
    
    log_steps = 0
    running_loss = 0
    running_loss_foresight = [0.0] * args.num_heads
    running_loss_primary = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            
            with torch.cuda.amp.autocast(dtype=ptdtype):
                _, zs_tilde, primary_loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)

            if train_steps >= warmup_steps_total:
                with torch.no_grad():
                    teacher_zs = ema_teacher.forward_ema(cond_idx=c_indices, idx=z_indices[:,:-1], targets=None)
                    zs_teacher_list = [teacher_zs]
                
                foresight_loss_list = calculate_foresight_loss(zs_teacher_list, zs_tilde)

                coeff_mult = get_piecewise_coeff_multiplier(epoch)
                current_proj_coeffs = [c * coeff_mult for c in args.proj_coeffs]


                total_foresight_loss = 0
                for i in range(args.num_heads):
                    total_foresight_loss += foresight_loss_list[i] * current_proj_coeffs[i]

                loss = primary_loss + total_foresight_loss
            else:
                fake_foresight_loss = 0.0
                for z in zs_tilde:
                    fake_foresight_loss += (z * 0).sum() 
                    
                loss = primary_loss + fake_foresight_loss
                foresight_loss_list = [torch.tensor(0.0, device=device) for _ in range(args.num_heads)]

            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)
                if args.use_prev_iter_ema:
                    ema_teacher.load_state_dict(ema.state_dict())

            running_loss += loss.item()
            for i in range(args.num_heads):
                running_loss_foresight[i] += foresight_loss_list[i].item()
            running_loss_primary += primary_loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_primary = torch.tensor(running_loss_primary / log_steps, device=device)
                avg_loss_foresight_heads = [torch.tensor(val / log_steps, device=device) for val in running_loss_foresight]

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_primary, op=dist.ReduceOp.SUM)
                for i in range(args.num_heads):
                    dist.all_reduce(avg_loss_foresight_heads[i], op=dist.ReduceOp.SUM)
                
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_primary = avg_loss_primary.item() / dist.get_world_size()
                avg_loss_foresight_heads = [val.item() / dist.get_world_size() for val in avg_loss_foresight_heads]

                log_msg = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Primary Loss: {avg_loss_primary:.4f}, "
                wandb_log = {
                    "total_loss": avg_loss,
                    "primary_loss": avg_loss_primary,
                    "steps_per_sec": steps_per_sec,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }   
                wandb_log["foresight_coeff_multiplier"] = get_piecewise_coeff_multiplier(epoch)
                for i in range(args.num_heads):
                    log_msg += f"Foresight Loss_{i+1}: {avg_loss_foresight_heads[i]:.4f}, "
                    wandb_log[f"foresight_loss_{i+1}"] = avg_loss_foresight_heads[i]
                log_msg += f"Train Steps/Sec: {steps_per_sec:.2f}"
                logger.info(log_msg)

                if rank == 0 and args.report_to_wandb:
                    wandb.log(wandb_log, step=train_steps)
                
                running_loss = 0
                running_loss_foresight = [0.0] * args.num_heads
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
    parser.add_argument("--dataset", type=str, required=True) 
    parser.add_argument("--proj-coeffs", type=float, nargs='+', default=[0.25, 0.25])
    parser.add_argument("--wandb-project", type=str, default="Mirai-E")
    parser.add_argument("--report-to-wandb", action='store_true')
    parser.add_argument("--use-prev-iter-ema", action='store_true')
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--num-heads", type=int, default=2) 
    parser.add_argument("--proj-coeff-decay-epoch", type=int, default=0,
                        help="Total steps for cosine decay of proj_coeffs. "
                             "Decay starts after warmup. If <= 0, no decay is applied.")

    args = parser.parse_args()
    main(args)