# Monocular Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps

This repository is the official PyTorch implementation of the paper:

&nbsp;&nbsp;[**Monocular Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps**](https://lynl7130.github.io/MonoDyGauBench.github.io/)  
[Yiqing Liang](https://lynl7130.github.io), [Mikhail Okunev](https://mmehas.github.io/), [Mikaela Angelina Uy](https://mikacuy.github.io/)†‡, [Runfeng Li](https://www.linkedin.com/in/runfeng-l-a41b6a204/), [Leonidas Guibas](https://profiles.stanford.edu/leonidas-guibas)‡, [James Tompkin](https://jamestompkin.com/), [Adam W Harley](https://adamharley.com/)‡  

<img width="12%"  text-align="center" margin="auto" src=images/brownlogo.svg> &nbsp;&nbsp;
†<img width="20%"  text-align="center" margin="auto" src=images/nvidia_logo.png>  &nbsp;&nbsp;
‡<img width="8%"  text-align="center" margin="auto" src=images/stanfordlogo.png>

&nbsp;&nbsp;&nbsp;[Paper](https://lynl7130.github.io/data/DyGauBench_tmp.pdf)

## Installation
This code has been developed with Anaconda (Python 3.7), CUDA 11.8.0 on Red Hat Enterprise Linux 9.2, one NVIDIA GeForce RTX 3090 GPU.  

```Shell
conda create -p [YourEnv] python=3.7
conda activate [YourEnv]

conda install -c anaconda libstdcxx-ng
conda install -c menpo opencv 

conda install -c conda-forge plyfile==0.8.1
pip install tqdm imageio

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install torchmetrics

pip install requests 

pip install tensorboard 

pip install scipy

pip install kornia



pip install lightning=2.2.1
pip install "jsonargparse[signatures]"

pip install wandb
pip install lpips

pip install pytorch-msssim

pip install ninja
pip install timm==0.4.5

# install from local folders 
pip install submodules/diff-gaussian-rasterization
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/gaussian-rasterization_ch3
pip install submodules/gaussian-rasterization_ch9
pip install submodules/simple-knn

```


After activating conda environment
```
wandb init

paste API key and create first project following instruction
```


## Usage


Default usage following PytorchLightning:

```python main.py [before] [subcommand] [after]```

As do not want to customize arguments by subcommand (fit/test), pass the config after the subcommand

```python main.py [subcommand] --config path/to/config.yaml```

Note: when building extension using pip install, have to run on the cluster where the pytorch is installed.
After building, can switch to different cluster.