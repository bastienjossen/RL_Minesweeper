#!/bin/bash
#SBATCH --job-name=det_DQN
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bastien.jossen@gmail.com

module load Anaconda3
module load CUDA/12.2.0
eval "$(conda shell.bash hook)"
conda activate dl_a2

python run.py
