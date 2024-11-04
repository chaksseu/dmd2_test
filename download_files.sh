export CHECKPOINT_PATH="/home/work/StableDiffusion/DMD2/ckpt_path" # change this to your own checkpoint folder 
export WANDB_ENTITY="rtrt505" # change this to your own wandb entity
export WANDB_PROJECT="DMD2_test" # change this to your own wandb project
export MASTER_IP="localhost"  # change this to your own master ip

# /etc/hosts 파일에 마스터 노드 정보 추가
echo "localhost    $(hostname)" | tee -a /etc/hosts

# 체크포인트 폴더 생성
mkdir -p $CHECKPOINT_PATH

# 모델 다운로드
bash scripts/download_sdv15.sh $CHECKPOINT_PATH
