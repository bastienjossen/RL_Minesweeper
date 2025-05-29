#!/bin/bash
#SBATCH --job-name=non_det_DQN_discount
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bastien.jossen@gmail.com

module load Anaconda3
module load CUDA/12.2.0
eval "$(conda shell.bash hook)"
conda activate dl_a2

python run.py
