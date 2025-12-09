Pre-extract discrete codes of training images
bash /home/y_yu/LlamaGen/scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --data-path /home/y_yu/data/ImageNet/train --code-path /home/y_yu/data/imagenet_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 256

bash /home/y_yu/LlamaGen/scripts/autoregressive/extract_codes_c2i_repa.sh --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --data-path /home/y_yu/data/ImageNet/train --code-path /home/y_yu/data/imagenet_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 256

Train AR models with DDP
bash /home/y_yu/LlamaGen/scripts/autoregressive/train_c2i.sh --cloud-save-path /home/y_yu/LlamaGen/result --code-path /home/y_yu/data/imagenet_code_c2i_flip_ten_crop --image-size 384 --gpt-model GPT-B
bash /home/y_yu/LlamaGen/scripts/autoregressive/train_c2i.sh --cloud-save-path /home/y_yu/LlamaGen/result --code-path /home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256 --image-size 256 --gpt-model GPT-B

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29500 \
 /home/y_yu/LlamaGen/autoregressive/train/train_c2i.py \
    --gpt-model="GPT-B" \
    --image-size=256 \
    --downsample-size=16 \
    --global-batch-size=256 \
    --lr=1e-4 \
    --epochs=300 \
    --ema \
    --results-dir="/home/y_yu/LlamaGen/results" \
    --cloud-save-path="/home/y_yu/LlamaGen/result" \
    --dataset="imagenet_code" \
    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" \
    --report-to-wandb


Sampling
bash scripts/autoregressive/sample_c2i.sh --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results/012-GPT-B_base_256/checkpoints/0050000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12342 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results/GPT-B_base_256/checkpoints/0300000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --ema

Evaluation
python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-XL-0600000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.75-seed-0-repa-coeff-0.5-Depth-4.npz
python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-0200000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0-repa-coeff-0.5-Depth-4.npz

accelerate launch /home/y_yu/LlamaGen/REPA/train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8" \
  --data-dir=/home/y_yu/data/imagenet_code_c2i_flip_ten_crop

python /home/y_yu/LlamaGen/REPA/preprocessing/dataset_tools.py convert --source=/home/y_yu/data/ImageNet/train/ --dest=/home/y_yu/data/repa/images --resolution=256x256 --transform=center-crop-dhariwal

python /home/y_yu/LlamaGen/REPA/preprocessing/dataset_tools.py encode --source=/home/y_yu/data/repa/images    --dest=/home/y_yu/data/repa/vae-sd


torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=12335 \
 /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa.py \
    --gpt-model="GPT-B" \
    --image-size=256 \
    --downsample-size=16 \
    --global-batch-size=256 \
    --lr=1e-4 \
    --epochs=300 \
    --ema \
    --results-dir="/home/y_yu/LlamaGen/results" \
    --cloud-save-path="/home/y_yu/LlamaGen/result" \
    --dataset="imagenet_repa" \
    --raw-image-path="/home/y_yu/data/ImageNet/train" \
    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" \
    --enc-type="dinov2-vit-b" \
    --encoder-depth=4 \
    --proj-coeff=0.5  \
    --json-path "/home/y_yu/LlamaGen/results" \

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
/home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py \
--vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
--gpt-ckpt /home/y_yu/LlamaGen/results/coeff_2_Depth_4_code_1_raw_image_1/checkpoints/0050000.pt \
--gpt-model GPT-B \
--image-size 256 \
--image-size-eval 256 \
--cfg-scale 2.0


torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12347 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results10/063-GPT-B/checkpoints/0200000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --ema


torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=12335 \
 /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa.py \
    --gpt-model="GPT-B" \
    --image-size=256 \
    --downsample-size=16 \
    --global-batch-size=256 \
    --lr=1e-4 \
    --epochs=300 \
    --ema \
    --results-dir="/home/y_yu/LlamaGen/results" \
    --cloud-save-path="/home/y_yu/LlamaGen/result" \
    --dataset="imagenet_repa" \
    --raw-image-path="/home/y_yu/data/ImageNet/train" \
    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop" \
    --enc-type="dinov2-vit-b" \
    --encoder-depth=4 \
    --proj-coeff=0.5  \
    --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop/256.json" \
    --report-to-wandb


torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12335  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop"     --enc-type="dinov2-vit-b"     --encoder-depth=4     --proj-coeff=4      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12345 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results/017-GPT-B/checkpoints/0050000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0  

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12345 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results5/010-GPT-B/checkpoints/0100000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8     --proj-coeff=2      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --gpt-ckpt /home/y_yu/LlamaGen/results/019-GPT-B_depth8_more/checkpoints/0100000.pt



torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12346 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results/018-GPT-B/checkpoints/0050000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0



python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-0300000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0-repa-coeff-0.5-Depth-4.npz




torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12335 \
 /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_multi.py \
    --gpt-model="GPT-B" \
    --image-size=256 \
    --downsample-size=16 \
    --global-batch-size=256 \
    --lr=1e-4 \
    --epochs=300 \
    --ema \
    --results-dir="/home/y_yu/LlamaGen/results" \
    --cloud-save-path="/home/y_yu/LlamaGen/result" \
    --dataset="imagenet_json" \
    --raw-image-path="/home/y_yu/data/ImageNet/train" \
    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" \
    --enc-type="dinov2-vit-b" \
    --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" \
    --report-to-wandb \
    --encoder-depths 4 8 \
    --proj-coeffs 1.0 1.0 \
    --gpt-ckpt "/home/y_yu/LlamaGen/results/019-GPT-B/checkpoints/0200000.pt" 

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12346 \
/home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa_multi.py \
    --gpt-model="GPT-B" \
    --gpt-ckpt "/home/y_yu/LlamaGen/results/017-GPT-B/checkpoints/0050000.pt" \
    --vq-ckpt "/home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt" \
    --image-size=256 \
    --image-size-eval=256 \
    --cfg-scale=2.0 \


   
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12346 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa_multi.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results/019-GPT-B/checkpoints/0300000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=2     --proj-coeff=2      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --use-prev-iter-ema

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8     --proj-coeff=0.5      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --use-prev-iter-ema  --ema-future-window-size 4

self：
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"   --encoder-depth=8     --proj-coeff=0.2      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --use-prev-iter-ema --ema-future-window-size 4 --warmup-steps 25000

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12345 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results5/017-GPT-B/checkpoints/0150000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0

python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-0050000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0-repa-coeff-0.5-Depth-4.npz

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12337  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=23     --ema     --results-dir="/home/y_yu/LlamaGen/results_1"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"   --encoder-depth=4     --proj-coeff=0.5   --gpt-ckpt /home/y_yu/LlamaGen/results_1/base_with_mlp/checkpoints/0075000.pt   --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --use-prev-iter-ema --ema-future-window-size 128 --warmup-steps 75000 --teacher-depth=8

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12338 /home/y_yu/LlamaGen/autoregressive/linear_probe.py     --gpt-ckpt "/home/y_yu/LlamaGen/results/GPT-B_base_256/checkpoints/0100000.pt"     --data-path "/home/y_yu/data/ImageNet"     --output-file "my_experiment_probe_results.json"     --gpt-model "GPT-B"     --image-size 256     --batch-size 128

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12338 /home/y_yu/LlamaGen/autoregressive/linear_probe.py     --gpt-ckpt "/home/y_yu/LlamaGen/results5/004-GPT-B_window_8_decay_0.4/checkpoints/0100000.pt"     --data-path "/home/y_yu/data/ImageNet"     --output-file "my_experiment_probe_results.json"     --gpt-model "GPT-B"     --image-size 256     --batch-size 256 --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt

torchrun --standalone --nproc_per_node=4 \
  /home/y_yu/LlamaGen/autoregressive/preprocess_tokenize.py \
  --data-path /home/y_yu/data/ImageNet \
  --output-path /home/y_yu/data/linear_probing \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --vq-model VQ-16 \
  --image-size 256 \
  --batch-size 256 \
  --num-workers 8


torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12338 /home/y_yu/LlamaGen/autoregressive/linear_probe.py     --gpt-ckpt "/home/y_yu/LlamaGen/results/GPT-B_base_256/checkpoints/0100000.pt"         --output-file "my_experiment_probe_results.json"     --gpt-model "GPT-B"      --batch-size 256 --cached-data-path /home/y_yu/data/linear_probing

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12338  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=50     --ema     --results-dir="/home/y_yu/LlamaGen/results_1"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"   --encoder-depth=8     --proj-coeff=0.5   --gpt-ckpt /home/y_yu/LlamaGen/results_1/base_with_mlp/checkpoints/0075000.pt   --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --use-prev-iter-ema --ema-future-window-size 8 --warmup-steps 75000 --teacher-depth=8

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_apart.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results3"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8     --proj-coeff=2      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_orgin_apart.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results3"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8     --proj-coeff=2      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --block-size=32

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results4"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --proj-coeffs 1.0 1.0 --gpt-ckpt /home/y_yu/LlamaGen/results4/006-GPT-B/checkpoints/0100000.pt --num-repa-heads=2


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results5"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=64 --future-ema-decay=0.4 --proj-coeff=2

torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results5"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.4 --proj-coeff=2


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results5"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=8 --future-ema-decay=0.4 --proj-coeff=2 --gpt-ckpt "/home/y_yu/LlamaGen/results5/004-GPT-B_window_8_decay_0.4/checkpoints/0100000.pt" 

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12345 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results9/006-GPT-L/checkpoints/0200000.pt --gpt-model GPT-L --image-size 256 --image-size-eval 256 --cfg-scale 2.0  --ema

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=32 --future-ema-decay=0.4 --proj-coeff=2 --use-prev-iter-ema


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12334  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results5"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.4 --proj-coeff=2 --gpt-ckpt "/home/y_yu/LlamaGen/results5/009-GPT-B__window_2_decay_0.4/checkpoints/0100000.pt" 


OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 
torchrun --nproc_per_node=8 train.py ... --omp-num-threads 8


OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results5"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=64 --future-ema-decay=0.4 --proj-coeff=2 --gpt-ckpt "/home/y_yu/LlamaGen/results5/012-GPT-B_window_64_decay_0.4/checkpoints/0100000.pt" 





OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa.py     --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8     --proj-coeff=2      --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --gpt-ckpt /home/y_yu/LlamaGen/results/019-GPT-B_depth8_more/checkpoints/0100000.pt --num-workers=10

OMP_NUM_THREAD=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6

OMP_NUM_THREAD=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=1 --use-prev-iter-ema --num-workers=4


OMP_NUM_THREAD=8 torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=1 --use-prev-iter-ema --num-workers=8

OMP_NUM_THREAD=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=0.8 --use-prev-iter-ema --num-workers=4

OMP_NUM_THREAD=6 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=0.8 --use-prev-iter-ema --num-workers=6

OMP_NUM_THREAD=8 torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=8


OMP_NUM_THREAD=5 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=5


OMP_NUM_THREAD=5 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=8 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=5

OMP_NUM_THREAD=5 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results6"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --proj-coeff=0.5 --use-prev-iter-ema --num-workers=5 --ema-future-window-size 0 --warmup-steps 0 --teacher-depth=8


OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future_dynamic_weight.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results7"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=32 --future-ema-decay=0.6 --proj-coeff=2 --num-workers=5


OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future_2D.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results7"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --proj-coeff=2 --num-workers=6

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12436  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future_2D.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results7"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --proj-coeff=2 --num-workers=5 --future-token=1

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future_2D.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results7"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --proj-coeff=2 --num-workers=5 --future-token=3

OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future_2D.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results7"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --proj-coeff=2 --num-workers=6 --future-token=1




CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results8"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=5 --gpt-ckpt /home/y_yu/LlamaGen/results_1/base_with_mlp/checkpoints/0075000.pt

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results8"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=5 --gpt-ckpt /home/y_yu/LlamaGen/results_1/base_with_mlp/checkpoints/0075000.pt


 OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results8"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.4 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=6 --gpt-ckpt /home/y_yu/LlamaGen/results_1/base_with_mlp/checkpoints/0075000.pt

  OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336    /home/y_yu/LlamaGen/autoregressive/train/train_c2i.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results9"     --cloud-save-path="/home/y_yu/LlamaGen/result9"     --dataset="imagenet_code"      --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"  --report-to-wandb --num-workers=6 

Linear~~~~

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results9"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=2 --num-workers=6

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results9"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6   torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12338  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results9"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results9/006-GPT-L/checkpoints/0075000.pt" 

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results9"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --num-workers=5

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12338 /home/y_yu/LlamaGen/autoregressive/linear_probe.py     --gpt-ckpt "/home/y_yu/LlamaGen/results9/001-GPT-L/checkpoints/0100000.pt"         --output-file "my_experiment_probe_results_L.json"     --gpt-model "GPT-L"      --batch-size 256 --cached-data-path /home/y_yu/data/linear_probing

CUDA_VISIBLE_DEVICES=4,5,6,7, torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12338 /home/y_yu/LlamaGen/autoregressive/linear_probe.py     --gpt-ckpt "/home/y_yu/LlamaGen/results9/001-GPT-L/checkpoints/0200000.pt"         --output-file "my_experiment_probe_results_L.json"     --gpt-model "GPT-L"      --batch-size 256 --cached-data-path /home/y_yu/data/linear_probing 

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=12339 /home/y_yu/LlamaGen/autoregressive/linear_probe_new.py     --gpt-ckpt "/home/y_yu/LlamaGen/results9/001-GPT-L/checkpoints/0200000.pt"         --output-file  "my_experiment_probe_results_L.json"    --gpt-model "GPT-L"  --batch-size 256 --cached-data-path /home/y_yu/data/linear_probing

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"    --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --num-workers=5 --proj-coeffs 0.25 0.25  --num-repa-heads=2 --warmup-steps 75000 --use-prev-iter-ema

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"    --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb  --proj-coeff=2 --num-workers=5 --proj-coeffs 0.25 0.25  --num-repa-heads=2 --warmup-steps 0 --use-prev-iter-ema




CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6   torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12339  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results9"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"    --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --num-workers=6 --proj-coeffs 0.25 0.25  --num-repa-heads=2 --warmup-steps 75000 --use-prev-iter-ema

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=5 --warmup-steps 75000

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"    --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --num-workers=5 --proj-coeffs 1 1  --num-repa-heads=2 --warmup-steps 75000 --use-prev-iter-ema --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B/checkpoints/0075000.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"    --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --num-workers=5 --proj-coeffs 100 100  --num-repa-heads=2 --warmup-steps 75000 --use-prev-iter-ema --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B/checkpoints/0075000.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=4  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=4 --warmup-steps 75000

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"


OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"

torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt" --num-workers=6


OMP_NUM_THREADS=6 torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt" --num-workers=6

OMP_NUM_THREADS=6 torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt" --num-workers=6


OMP_NUM_THREADS=6 torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results_1/base_with_mlp/checkpoints/0075000.pt" --num-workers=6

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4  --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results9"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --num-workers=5

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"

CUDA_VISIBLE_DEVICES=4,5,6,7, torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"


 torchrun --nnodes=1 --nproc_per_node=1   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"

 CUDA_VISIBLE_DEVICES=4,5,6,7, torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=8 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"

 OMP_NUM_THREADS=6  CUDA_VISIBLE_DEVICES=4,5,6,7,  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"

 CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=8 --warmup-steps 0

 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12336  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results5"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"     --enc-type="dinov2-vit-b"     --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=32 --future-ema-decay=0.6 --proj-coeff=2 --num-workers=6 --gpt-ckpt /home/y_yu/LlamaGen/results5/022-GPT-B_window_32_decay_0.6/checkpoints/0050000.pt

  OMP_NUM_THREADS=6  CUDA_VISIBLE_DEVICES=4,5,6,7,  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12337   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=16 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"


CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=5 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"

OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=24 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=43     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=2 --future-ema-decay=0.6 --proj-coeff=0.5 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B_one_head_base/checkpoints/0075000.pt"

OMP_NUM_THREADS=1  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=5 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B/checkpoints/0075000.pt"



OMP_NUM_THREADS=1  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb  --proj-coeffs 1 1 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt" --num-repa-heads=2


--gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"    --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --num-workers=5 --proj-coeffs 100 100  --num-repa-heads=2 --warmup-steps 75000 --use-prev-iter-ema --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt"

OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --encoder-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb  --proj-coeffs 1 1 --use-prev-iter-ema --num-workers=5 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt" --num-repa-heads=2

OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=4     --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb  --proj-coeffs 1 1 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt" --num-repa-heads=2

python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-0100000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0-repa-coeff-0.5-Depth-4.npz

OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346  /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_self_one_head_future.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8        --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb --future-window-size=1 --future-ema-decay=0.6 --proj-coeff=2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/011-GPT-B_one_head_base/checkpoints/0075000.pt"

OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=4     --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb  --proj-coeffs 1 1 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt" --num-repa-heads=2

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12347   /home/y_yu/Mirai/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=5 --warmup-steps 75000  --num-repa-heads=3

CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"     --raw-image-path="/home/y_yu/data/ImageNet/train"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --json-path "/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json" --report-to-wandb  --proj-coeffs 2 2 --use-prev-iter-ema --num-workers=5 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt" --num-repa-heads=2

 CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_two_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"       --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000 --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt" --num-repa-heads=2


 CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12347   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results10"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"     --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000  --num-repa-heads=3

     --dataset="imagenet_code" \
    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" \


 CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=45     --ema     --results-dir="/home/y_yu/LlamaGen/results11"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000  --num-repa-heads=1



 CUDA_VISIBLE_DEVICES=0,1,2,3,  OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=43     --ema     --results-dir="/home/y_yu/LlamaGen/results11"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"       --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 3 3  --use-prev-iter-ema --num-workers=6 --warmup-steps 75000  --num-repa-heads=2 --gpt-ckpt "/home/y_yu/LlamaGen/results10/027-GPT-B_two_head_base/checkpoints/0075000.pt"

 CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=42     --ema     --results-dir="/home/y_yu/LlamaGen/results11"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 3 3 3 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results11/001-GPT-B_threehead_2_2_2/checkpoints/0075000.pt" --ckpt-every 200000

 CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12349   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results11"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 1 1 1 1 --use-prev-iter-ema --num-workers=6 --warmup-steps 75000  --num-repa-heads=4 --gpt-ckpt "/home/y_yu/LlamaGen/results11/002-GPT-B_4head_2_2_2_2/checkpoints/0075000.pt" --ckpt-every 100000


 CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=1024     --lr=4e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results13"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=6 --warmup-epochs 15  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results11/001-GPT-B_threehead_2_2_2/checkpoints/0075000.pt" --ckpt-every 25000 --proj-coeff-decay-epoch 285


 CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12347    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=1024     --lr=4e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results13"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=8   --student-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 2 2 2  --num-workers=6 --warmup-epochs 15  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results13/012-GPT-B/checkpoints/0150000.pt" --ckpt-every 25000 --proj-coeff-decay-epoch 285



CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=1024     --lr=4e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results13"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=6 --warmup-epochs 15  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results13/014-GPT-B/checkpoints/0175000.pt" --ckpt-every 25000 --proj-coeff-decay-epoch 285

CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12347    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results14"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=8   --student-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 1 1 1  --num-workers=6 --warmup-epochs 0  --num-repa-heads=3  --ckpt-every 100000 --proj-coeff-decay-epoch 0

 CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results14"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=6 --warmup-epochs 15  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results14/001-GPT-B/checkpoints/0800000.pt" --ckpt-every 100000 --proj-coeff-decay-epoch 285



 torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=12348 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results14/002-GPT-B_ema_2_2_2_decay/checkpoints/1500000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --ema

python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-1500000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0-repa-coeff-0.5-Depth-4.npz

 CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results14"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 1 1 1 --use-prev-iter-ema --num-workers=6 --warmup-epochs 15  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results14/001-GPT-B/checkpoints/0800000.pt" --ckpt-every 100000 --proj-coeff-decay-epoch 0


 CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12347    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_L"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=16   --student-depth=16    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 2  --num-workers=6 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 --proj-coeff-decay-epoch 0

  CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_L"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=13   --student-depth=13    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 2  --num-workers=6 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 --proj-coeff-decay-epoch 0


  CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6 torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29500 \
 /home/y_yu/LlamaGen/autoregressive/train/train_c2i.py \
    --gpt-model="GPT-B" \
    --image-size=256 \
    --downsample-size=16 \
    --global-batch-size=256 \
    --lr=1e-4 \
    --epochs=300 \
    --ema \
    --results-dir="/home/y_yu/LlamaGen/results_baseline" \
    --cloud-save-path="/home/y_yu/LlamaGen/result" \
    --dataset="imagenet_code" \
    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" \
    --report-to-wandb \
    --num-workers=6 \
    --ckpt-every 100000

  CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results14"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 1 1 1 --use-prev-iter-ema --num-workers=6 --warmup-epochs 15  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results14/001-GPT-B/checkpoints/0800000.pt" --ckpt-every 100000 --proj-coeff-decay-epoch 0

   CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results14"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 0.5 0.5 0.5 --use-prev-iter-ema --num-workers=6 --warmup-epochs 15  --num-repa-heads=3 --gpt-ckpt "/home/y_yu/LlamaGen/results14/007-GPT-B__ema_2_2_2_160epoch_2changeto1/checkpoints/1400000.pt" --ckpt-every 100000 --proj-coeff-decay-epoch 0

   torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12342 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results_L/000-GPT-L/checkpoints/0700000.pt --gpt-model GPT-L --image-size 256 --image-size-eval 256 --cfg-scale 2 --ema --per-proc-batch-size 64


OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train_new/train_c2i_repa_three_head_self_change_coef_XL.py    --gpt-model="GPT-XL"     --image-size=384     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_XL"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_384"      --teacher-depth=24   --student-depth=24     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=3  --ckpt-every 100000 





OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_L"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=16   --student-depth=16    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 1  --num-workers=6 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 --proj-coeff-decay-epoch 0 --gpt-ckpt "/home/y_yu/LlamaGen/results_L/000-GPT-L/checkpoints/0800000.pt" 


<!-- OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-L"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_L"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=16   --student-depth=16    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 1  --num-workers=6 --warmup-epochs 300  --num-repa-heads=1  --ckpt-every 100000 --proj-coeff-decay-epoch 0 --gpt-ckpt "/home/y_yu/LlamaGen/results_L/000-GPT-L/checkpoints/0600000.pt"  -->


OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train_new/train_c2i_repa_mask.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_L"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --encoder-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeff 2  --num-workers=6   --ckpt-every 100000


   CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train_new/train_c2i_repa_mask.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_L"     --cloud-save-path="/home/y_yu/LlamaGen/result_mask"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --encoder-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeff 2  --num-workers=6   --ckpt-every 50000

      CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345    /home/y_yu/LlamaGen/autoregressive/train_new/train_c2i_repa_mask.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_L"     --cloud-save-path="/home/y_yu/LlamaGen/result_mask"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --encoder-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeff 4  --num-workers=6   --ckpt-every 50000

      
OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i.py    --gpt-model="GPT-XL"     --image-size=256  --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_XL"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_384" --report-to-wandb --num-workers=8  --ckpt-every 100000 




  torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12342 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt /home/y_yu/LlamaGen/results_L/000-GPT-L/checkpoints/0700000.pt --gpt-model GPT-L --image-size 256 --image-size-eval 256 --cfg-scale 2 --ema --per-proc-batch-size 64


   torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12348 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_XL/005-GPT-XL/checkpoints/0200000.pt --gpt-model GPT-XL --image-size 256 --image-size-eval 256 --cfg-scale 1.75 --ema --precision none

    torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12348 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_XL/005-GPT-XL/checkpoints/0200000.pt --gpt-model GPT-XL --image-size 256 --image-size-eval 256 --cfg-scale 1.75 --ema --precision fp16 --per-proc-batch-size 32

    torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12348 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_XL/005-GPT-XL/checkpoints/0600000.pt --gpt-model GPT-XL --image-size 256 --image-size-eval 256 --cfg-scale 1.75 --ema --per-proc-batch-size 64

##########
  CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_no_change.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_coeff"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=8   --student-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 3  --num-workers=16 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 --proj-coeff-decay-epoch 0 --schedual="always_3"  --gpt-ckpt "/home/y_yu/LlamaGen/results_coeff/003-GPT-B/checkpoints/0600000.pt"

  CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12349    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_coeff"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=8   --student-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 2  --num-workers=16 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 --proj-coeff-decay-epoch 0 --gpt-ckpt "/home/y_yu/LlamaGen/results_coeff/009-GPT-B/checkpoints/1200000.pt"

  CUDA_VISIBLE_DEVICES=4,5,6,7, OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_coeff_cos.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_coeff"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=8   --student-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 2  --num-workers=8 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 --proj-coeff-decay-epoch 300 --schedual="cos2-0"  

     CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=5  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_Depth"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=4   --student-depth=4     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=5 --warmup-epochs 15  --num-repa-heads=3 --ckpt-every 100000 --proj-coeff-decay-epoch 0 --gpt-ckpt "/home/y_yu/LlamaGen/results_Depth/ema_3heads_baseline/checkpoints/0075000.pt"


OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i.py    --gpt-model="GPT-XL"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_XL"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" --report-to-wandb --num-workers=8  --ckpt-every 100000 --gpt-ckpt "/home/y_yu/LlamaGen/results12/000-GPT-B_dino_2/checkpoints/0800000.pt"

   torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12349 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_coeff/019-GPT-B_cos_2_down_0/checkpoints/1500000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2 --ema

   python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-1500000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0-repa-coeff-0.5-Depth-4.npz

   OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_baseline_lr_decrease.py    --gpt-model="GPT-XL"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_XL"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" --report-to-wandb --num-workers=8  --ckpt-every 100000 --gpt-ckpt "/home/y_yu/LlamaGen/results_XL/012-GPT-XL/checkpoints/1000000.pt"


      CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_baseline_lr_decrease.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_baseline"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" --report-to-wandb --num-workers=8  --ckpt-every 100000 --gpt-ckpt "/home/y_yu/LlamaGen/results_baseline/001-GPT-B/checkpoints/1000000.pt"


      

 OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_warmup"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 0  --num-repa-heads=3 --ckpt-every 100000    


CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=4   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train_new/train_c2i_repa_three_head_self_change_coef_lr_decay.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_Decay"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=3 --ckpt-every 100000 --gpt-ckpt "/home/y_yu/LlamaGen/results14/007-GPT-B__ema_2_2_2_160epoch_2changeto1_best/checkpoints/1000000.pt"

 OMP_NUM_THREADS=1  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=1500     --ema     --results-dir="/home/y_yu/LlamaGen/results_baseline"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256" --report-to-wandb --num-workers=24  --ckpt-every 100000 --gpt-ckpt "/home/y_yu/LlamaGen/results_baseline/002-GPT-B/checkpoints/1000000.pt"


python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results12/000-GPT-B_dino_2/checkpoints/1400000.pt --gpt-model GPT-B --image-size 256  --cfg-scale 2 --ema   --class-id 292  --num-samples 16 --seed 0 --train-method "dino"

python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results14/007-GPT-B__ema_2_2_2_160epoch_2changeto1_best/checkpoints/1400000.pt --gpt-model GPT-B --image-size 256  --cfg-scale 2 --ema   --class-id 292  --num-samples 16 --seed 0 --train-method "ema"

python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1400000.pt --gpt-model GPT-B --image-size 256  --cfg-scale 2 --ema   --class-id 292  --num-samples 16 --seed 0 --train-method "base"


python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_XL/005-GPT-XL_baseline/checkpoints/0300000.pt --gpt-model GPT-XL --image-size 256  --cfg-scale 1.75 --ema   --class-id 972 --num-samples 16 --seed 42 --train-method "base"

 OMP_NUM_THREADS=6  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_warmup"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=6 --warmup-epochs 0  --num-repa-heads=3 --ckpt-every 100000   

#####
python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_miyabi/result_XL_DINO/1500000.pt --gpt-model GPT-XL --image-size 256  --cfg-scale 1.75 --ema   --class-id 207 --num-samples 100 --seed 0 --train-method "dino"

python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_miyabi/result_XL_SELF/1500000.pt --gpt-model GPT-XL --image-size 256  --cfg-scale 1.75 --ema   --class-id 207 --num-samples 100 --seed 42 --train-method "ema"

python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_miyabi/result_XL_DINO/1500000.pt --gpt-model GPT-XL --image-size 256  --cfg-scale 1.75 --ema   --class-id 817 --num-samples 100 --seed 42 --train-method "dino"

python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_miyabi/result_XL_SELF/1500000.pt --gpt-model GPT-XL --image-size 256  --cfg-scale 1.75 --ema   --class-id 817 --num-samples 100 --seed 0 --train-method "ema"

python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_miyabi/result_XL_DINO/1500000.pt --gpt-model GPT-XL --image-size 256  --cfg-scale 1.75 --ema   --class-id 975 --num-samples 120 --seed 0 --train-method "dino"

python /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_miyabi/result_XL_SELF/1500000.pt --gpt-model GPT-XL --image-size 256  --cfg-scale 1.75 --ema   --class-id 975 --num-samples 120 --seed 0 --train-method "ema"

#####
 python /home/y_yu/LlamaGen/autoregressive/tsne/tsne.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 284 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1000000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result


 python /home/y_yu/LlamaGen/autoregressive/tsne/tsne.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 284 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results12/000-GPT-B_dino_2/checkpoints/1500000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

   python /home/y_yu/LlamaGen/autoregressive/tsne/tsne.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 284 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results14/007-GPT-B__ema_2_2_2_160epoch_2changeto1_best/checkpoints/1500000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

###############
 python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 340 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1500000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

###############

   python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 340 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1400000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result


   python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 340 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results14/007-GPT-B__ema_2_2_2_160epoch_2changeto1_best/checkpoints/1400000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result


     python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 340 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results12/000-GPT-B_dino_2/checkpoints/1400000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

   python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png \
  --label 340 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt   /home/y_yu/LlamaGen/results_miyabi/result_XL_DINO/1500000.pt \
  --gpt-model GPT-XL \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result
###
     python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/paper/image.png \
  --label 284 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results14/007-GPT-B__ema_2_2_2_160epoch_2changeto1_best/checkpoints/1500000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

       python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/paper/image.png \
  --label 284 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1200000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result


       python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/paper/image.png \
  --label 284 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results12/000-GPT-B_dino_2/checkpoints/1200000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result
###

       python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/rocket.png \
  --label 812 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results12/000-GPT-B_dino_2/checkpoints/1300000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result


         python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/rocket.png \
  --label 812 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1200000.pt  \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

           python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/rocket.png \
  --label 812 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results14/007-GPT-B__ema_2_2_2_160epoch_2changeto1_best/checkpoints/1300000.pt  \
  --gpt-model GPT-B \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

             python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_new.py \
  --image /home/y_yu/LlamaGen/autoregressive/tsne/rocket.png \
  --label 812 \
  --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt /home/y_yu/LlamaGen/results_XL/013-GPT-XL/checkpoints/1500000.pt  \
  --gpt-model GPT-XL \
  --image-size 256 \
  --downsample-size 16 \
  --cls-token-num 0 \
  --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result



  ####

  python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_1D.py   --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png   --label 340   --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt   --gpt-ckpt /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1500000.pt   --gpt-model GPT-B   --image-size 256   --downsample-size 16   --cls-token-num 0   --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result
  

  python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_1D.py   --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png   --label 340   --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt   --gpt-ckpt /home/y_yu/LlamaGen/results_baseline/006-GPT-B/checkpoints/1400000.pt   --gpt-model GPT-B   --image-size 256   --downsample-size 16   --cls-token-num 0   --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

  python /home/y_yu/LlamaGen/autoregressive/tsne/tsne_1D.py   --image /home/y_yu/LlamaGen/autoregressive/tsne/zebra_full_256.png   --label 340   --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt   --gpt-ckpt /home/y_yu/LlamaGen/results12/000-GPT-B_dino_2/checkpoints/1500000.pt   --gpt-model GPT-B   --image-size 256   --downsample-size 16   --cls-token-num 0   --output-dir /home/y_yu/LlamaGen/autoregressive/tsne/result

 ####

 OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train_new/train_c2i_repa_three_head_self_384.py    --gpt-model="GPT-B"     --image-size=384     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_384"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_384"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=3 --ckpt-every 100000 

OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12348    /home/y_yu/LlamaGen/autoregressive/train_new/train_c2i_repa_three_head_384.py    --gpt-model="GPT-B"     --image-size=384     --downsample-size=16  --resolution 384   --global-batch-size=256    --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_384"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_384"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "//home/y_yu/data/imagenet_code_c2i_flip_ten_crop_384/imagenet384_manifest.json"  --teacher-depth=8   --student-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 2  --num-workers=8 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 --resolution 384

 OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/LlamaGen/autoregressive/train/train_c2i.py    --gpt-model="GPT-B"     --image-size=384     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_384"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_384" --report-to-wandb --num-workers=8  --ckpt-every 100000 

    torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12349 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2 --ema --gpt-ckpt /home/y_yu/LlamaGen/results_warmup/000-GPT-B/checkpoints/0400000.pt


torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12349 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-model GPT-B --image-size 384 --image-size-eval 256 --cfg-scale 2 --ema --gpt-ckpt /home/y_yu/LlamaGen/results_384/001-GPT-B——ema/checkpoints/0400000.pt

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12349 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-model GPT-B --image-size 384 --image-size-eval 256 --cfg-scale 2 --ema --gpt-ckpt /home/y_yu/LlamaGen/results_384/009-GPT-B——dino/checkpoints/0400000.pt

python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-0400000-size-384-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0-repa-coeff-0.5-Depth-4.npz

######

 OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_self_aroud_avg.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_L_around"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=3 --ckpt-every 100000   


  OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_self_aroud_avg.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_L_around"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 2 2 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=6 --ckpt-every 100000   --gpt-ckpt "/home/y_yu/LlamaGen/results_Depth/ema_3heads_baseline/checkpoints/0075000.pt" 


# 2 1 0.5

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12349 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp_repa.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_VAE/001-GPT-B/checkpoints/0100000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2 --ema

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12349 /home/y_yu/LlamaGen/autoregressive/sample/sample_c2i_ddp.py --vq-ckpt /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt --gpt-ckpt  /home/y_yu/LlamaGen/results_semantic_close_AdaLN_Bilateral_filter_content_new/002-GPT-B-Idea1-Dynamic/checkpoints/0400000.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2 --ema

python3 evaluations/c2i/evaluator.py /home/y_yu/LlamaGen/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz /home/y_yu/LlamaGen/samples/GPT-B-0400000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz

######
OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_VAE.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_VAE"     --cloud-save-path="/home/y_yu/LlamaGen/result"      --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"     --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2  --num-workers=8 --warmup-epochs 0  --num-repa-heads=3 --ckpt-every 100000  --vqgan-ckpt-path /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt


OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12346   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_VAE.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_VAE"     --cloud-save-path="/home/y_yu/LlamaGen/result"      --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet_wang/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"     --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2  --num-workers=8 --warmup-epochs 0  --num-repa-heads=1 --ckpt-every 100000  --vqgan-ckpt-path /home/y_yu/LlamaGen/pretrained_models/vq_ds16_c2i.pt



####
OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_semantic_close_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_semantic_close"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 2 2 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=6 --ckpt-every 100000  --gpt-ckpt "/home/y_yu/LlamaGen/results_semantic_close_AdaLN/000-GPT-B-Idea1-Dynamic_foresight_3/checkpoints/0075000.pt" 


####

OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_semantic_close_self_AdaLN.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_semantic_close_AdaLN"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 1 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=6 --ckpt-every 100000  --gpt-ckpt "/home/y_yu/LlamaGen/results_semantic_close_AdaLN/002-GPT-B-Idea1-Dynamic_foresight_6_coeff1/checkpoints/0100000.pt" 


OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_semantic_close_self_AdaLN_Bilateral_filter.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_semantic_close_AdaLN_Bilateral_filter"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 1 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=6 --ckpt-every 100000 --sigma-spatial 3 --sigma-range 0.5 --gpt-ckpt "/home/y_yu/LlamaGen/results_semantic_close_AdaLN_Bilateral_filter_content_new/000-GPT-B-Idea1-Dynamic/checkpoints/0075000.pt" 


OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_semantic_close_self_AdaLN_Bilateral_filter_content.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_semantic_close_AdaLN_Bilateral_filter_content_new"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=3 --ckpt-every 100000 --gpt-ckpt "/home/y_yu/LlamaGen/results_semantic_close_AdaLN_Bilateral_filter_content_new/000-GPT-B-Idea1-Dynamic/checkpoints/0075000.pt" 


OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8   --node_rank=0 --master_addr=127.0.0.1 --master_port=12344   /home/y_yu/LlamaGen/autoregressive/train_second/train_c2i_repa_next_hidden_state_prediction.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_next_hidden_prediction"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 --use-prev-iter-ema --num-workers=8 --warmup-epochs 15  --num-repa-heads=1 --ckpt-every 25000  --gpt-ckpt "/home/y_yu/LlamaGen/results_next_hidden_prediction/000-GPT-B/checkpoints/0075000.pt" 

###
OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8  --node_rank=0 --master_addr=127.0.0.1 --master_port=12345   /home/y_yu/Mirai/autoregressive/train/train_c2i_repa_three_head_self.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16     --global-batch-size=256     --lr=1e-4     --epochs=80     --ema     --results-dir="/home/y_yu/LlamaGen/results_warmup"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_code"    --code-path="/home/y_yu/data/imagenet_code_c2i_flip_ten_crop_256"      --teacher-depth=8   --student-depth=8     --report-to-wandb  --proj-coeffs 2 2 2 --use-prev-iter-ema --num-workers=6 --warmup-epochs 0  --num-repa-heads=3 --ckpt-every 100000   



OMP_NUM_THREADS=8  torchrun --nnodes=1 --nproc_per_node=8    --node_rank=0 --master_addr=127.0.0.1 --master_port=12349    /home/y_yu/Mirai/autoregressive/train/train_c2i_repa_three_head.py    --gpt-model="GPT-B"     --image-size=256     --downsample-size=16  --resolution 256   --global-batch-size=256    --lr=1e-4     --epochs=300     --ema     --results-dir="/home/y_yu/LlamaGen/results_coeff"     --cloud-save-path="/home/y_yu/LlamaGen/result"     --dataset="imagenet_json"    --code-path="/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256"   --raw-image-path="/home/y_yu/data/ImageNet/train"   --json-path "/home/ze_wang/yu/imagenet_code_c2i_flip_ten_crop_256/imagenet_256_manifest.json"  --teacher-depth=8   --student-depth=8    --enc-type="dinov2-vit-b"  --report-to-wandb  --proj-coeffs 2  --num-workers=8 --warmup-epochs 0  --num-repa-heads=1  --ckpt-every 100000 

###