#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH --partition=orion --gres=gpu:a4000:1
#SBATCH --job-name install
#SBATCH --output install.out
#SBATCH --account=orion
#SBATCH --nodelist=oriong12

#SBATCH --job-name hypernerf_split-cookie_TRBF_nodecoder1
#SBATCH --output nodecoder1.out

base="hypernerf/split-cookie/TRBF"
name="nodecoder1"
variant="${base}/${name%?}1"
output_path="./output/${base}"


chmod -R 777 $current_path/${name}.out



source /orion/u/yiqingl/anaconda3/etc/profile.d/conda.sh

cd /orion/u/yiqingl/GaussianDiff

#conda create -p /orion/u/yiqingl/envs/gaufre python=3.9
conda activate /orion/u/yiqingl/envs/gaufre
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}$
#export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/orion/u/yiqingl/.cache/torch_extension

nvcc --version
nvidia-smi



which python 
which pip
#pip install matplotlib

#pip install imageio[ffmpeg]
#pip install imageio[av]

#pip install submodules/diff-gaussian-rasterization
#pip install submodules/depth-diff-gaussian-rasterization
#pip install submodules/gaussian_rasterization_ch3
#pip install submodules/gaussian_rasterization_ch9
#pip install submodules/simple-knn

python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"



python -c "import torch; print(torch.backends.cudnn.version())"


#ls /usr/local/cuda-11.8/lib64/

cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
find /orion/u/yiqingl/envs/gaufre -name "libcudnn*"

variant="hypernerf/split-cookie/TRBF/nodecoder1"

python main.py fit --config configs/${variant}.yaml
python main.py test --config configs/${variant}.yaml  --ckpt_path  last #--print_config #--trainer.strategy FSDP #--print_config

rm -rf output/${variant}/wandb

chmod -R 777 $current_path/${name}.out

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW



