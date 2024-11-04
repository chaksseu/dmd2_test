export CHECKPOINT_PATH="/home/work/StableDiffusion/DMD2/ckpt_path" # change this to your own checkpoint folder 
export WANDB_ENTITY="rtrt505" # change this to your own wandb entity
export WANDB_PROJECT="DMD2_test" # change this to your own wandb project
export MASTER_IP="localhost"  # change this to your own master ip
export NUM_GPUS=2
export model_name="CompVis/stable-diffusion-v1-4"

python main/test_folder_sd.py   --folder $CHECKPOINT_PATH/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch/time_1730605623_seed10 \
    --wandb_name test_laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --image_resolution 512 \
    --latent_resolution 64 \
    --num_train_timesteps 1000 \
    --test_visual_batch_size 64 \
    --per_image_object 16 \
    --seed 10 \
    --anno_path $CHECKPOINT_PATH/captions_coco14_test.pkl \
    --eval_res 256 \
    --ref_dir $CHECKPOINT_PATH/val2014 \
    --total_eval_samples 30000 \
    --model_id $model_name\
    --pred_eps \
    --clip_score \