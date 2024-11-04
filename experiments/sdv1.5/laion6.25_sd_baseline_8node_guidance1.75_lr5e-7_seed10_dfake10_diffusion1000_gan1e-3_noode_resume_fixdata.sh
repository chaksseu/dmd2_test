export CHECKPOINT_PATH="/home/work/StableDiffusion/DMD2/ckpt_path" # change this to your own checkpoint folder 
export WANDB_ENTITY="rtrt505" # change this to your own wandb entity
export WANDB_PROJECT="DMD2_test" # change this to your own wandb project
export MASTER_IP="localhost"  # change this to your own master ip
export NUM_GPUS=2
#export model_name="CompVis/stable-diffusion-v1-4"
#"/home/work/StableDiffusion/DMD2/checkpoint-300000"
export model_name="CompVis/stable-diffusion-v1-4"
export batch_size=16

torchrun --nnodes 1 --nproc_per_node=$NUM_GPUS --rdzv_id=2345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP main/train_sd.py \
    --generator_lr 5e-7  \
    --guidance_lr 5e-7 \
    --train_iters 100000000 \
    --output_path $CHECKPOINT_PATH/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume \
    --batch_size $batch_size \
    --grid_size 2 \
    --initialie_generator --log_iters 1000 \
    --resolution 512 \
    --latent_resolution 64 \
    --seed 10 \
    --real_guidance_scale 1.75 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id $model_name \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --wandb_iters 50 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_name "stage2_only_channel_small_original_300000_kt_batch32"  \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 10 \
    --gradient_checkpointing \
    --cls_on_clean_image \
    --gen_cls_loss \
    --gen_cls_loss_weight 1e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --ckpt_only_path $CHECKPOINT_PATH/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000" \
    --real_image_path $CHECKPOINT_PATH/sensei-fs/users/tyin/cvpr_data/sd_vae_latents_laion_500k_lmdb/ \
    --num_workers=12 \
