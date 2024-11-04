export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3

# 사용하려는 GPU 수를 지정합니다. batch 원래32
export NUM_GPUS=2

torchrun --nnodes 1 --nproc_per_node=$NUM_GPUS --rdzv_id=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 main/train_sd.py \
    --generator_lr 1e-5 \
    --guidance_lr 1e-5 \
    --train_iters 100000000 \
    --output_path $CHECKPOINT_PATH/laion6.25_sd_baseline_${NUM_GPUS}gpu_guidance1.75_lr1e-5_seed10_dfake10_from_scratch \
    --batch_size 32 \
    --grid_size 2 \
    --initialie_generator --log_iters 1000 \
    --resolution 512 \
    --latent_resolution 64 \
    --seed 10 \
    --real_guidance_scale 1.75 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --wandb_iters 50 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_name "laion6.25_sd_baseline_${NUM_GPUS}gpu_guidance1.75_lr1e-5_seed10_dfake10_from_scratch" \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 10 \
    --gradient_checkpointing \
    --real_image_path $CHECKPOINT_PATH/sensei-fs/users/tyin/cvpr_data/sd_vae_latents_laion_500k_lmdb/ \