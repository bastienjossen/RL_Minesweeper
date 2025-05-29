#!/bin/bash
#SBATCH --job-name=visualize_progress
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bastien.jossen@gmail.com

module load Anaconda3
module load CUDA/12.2.0
eval "$(conda shell.bash hook)"
conda activate dl_a2

python vizualize.py
