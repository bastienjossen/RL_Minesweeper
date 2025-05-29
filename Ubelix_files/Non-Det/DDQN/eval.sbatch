#!/bin/bash
#SBATCH --job-name=eval_non_det_DQN
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bastien.jossen@gmail.com

module load Anaconda3
module load CUDA/12.2.0
eval "$(conda shell.bash hook)"
conda activate dl_a2

python evaluate.py --ckpt best_3.pth --episodes 1000 --out_csv eval_best_3_results.csv --nameP DQN_best_3
