#!/bin/bash
#SBATCH --job-name=acc
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda/3.7
source activate /opt/ohpc/pub/apps/pytorch_1.10
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2

stdbuf -oL -eL python3.7 -u main.py
