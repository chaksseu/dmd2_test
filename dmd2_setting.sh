sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install wget git zip unzip -y
conda create -n dmd2 python=3.8 -y 
conda activate dmd2 
pip install --upgrade pip
pip install --upgrade anyio
pip install -r requirements.txt
pip install -r my_requirements.txt
python setup.py develop
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
