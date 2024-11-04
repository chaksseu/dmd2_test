export CHECKPOINT_PATH="/home/work/StableDiffusion/DMD2/ckpt_path" # change this to your own checkpoint folder 
export WANDB_ENTITY="rtrt505" # change this to your own wandb entity
export WANDB_PROJECT="DMD2_test" # change this to your own wandb project
export MASTER_IP="localhost"  # change this to your own master ip

# start a training with 2 gpu. we need to run this script on only 1 nodes. 
bash experiments/sdv1.5/one_server_multi_gpu.sh $CHECKPOINT_PATH  $WANDB_ENTITY $WANDB_PROJECT $MASTER_IP

# on some other machine, start a testing process that continually reads from the checkpoint folder and evaluate the FID 
# Change TIMESTAMP_TBD to the real one
python main/test_folder_sd.py   --folder $CHECKPOINT_PATH/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch/TIMESTAMP_TBD \
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
    --model_id "runwayml/stable-diffusion-v1-5" \
    --pred_eps 