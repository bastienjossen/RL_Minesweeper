#!/bin/bash
#SBATCH --job-name=visualize_progress
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bastien.jossen@gmail.com

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dl_a2

python visualize_w_legal_mask.py
