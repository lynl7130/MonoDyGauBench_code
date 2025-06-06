#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=3090-gcondo --gres=gpu:2
#SBATCH --job-name debug_effgs_after4DGS
#SBATCH --output debug_effgs_after4DGS.out


module load gcc/10.1.0-mojgbn
module load cmake/3.26.3-xi6h36u
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load glm/0.9.9.8-m3s6sze

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh


cd ~/data/yliang51/GaussianDiff

conda -V

#conda remove -p ~/data/yliang51/envs/gaufre --all
#conda create -p ~/data/yliang51/envs/gaufre python=3.7
conda activate ~/data/yliang51/envs/gaufre

which python 
which pip


#pip uninstall -y lightning
#pip install lightning==2.2.1
#pip install "jsonargparse[signatures]"

#pip install tensorboard
#pip install wandb

#pip install submodules/depth-diff-gaussian-rasterization
#pip install submodules/gaussian_rasterization_ch3
#pip install submodules/gaussian_rasterization_ch9
#pip install kornia
#pip install pytorch-msssim


rm output/base_checkpoint_effgs/config.yaml
#pip install jsonargparse
python main.py fit --config configs/base_checkpoint_effgs.yaml
python main.py test --config configs/base_checkpoint_effgs.yaml  --ckpt_path  last #--print_config #--trainer.strategy FSDP #--print_config

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW



