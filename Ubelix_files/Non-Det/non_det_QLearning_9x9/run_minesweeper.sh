#!/bin/bash
#SBATCH --job-name=Q-Learning_non_det
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bastien.jossen@gmail.com

source ~/.bashers
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dl_a2

python train.py
