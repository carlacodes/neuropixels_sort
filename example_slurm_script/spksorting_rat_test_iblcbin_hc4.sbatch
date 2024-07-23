#!/bin/bash
#SBATCH --job-name=hc4rat_test_23072024_2
#SBATCH --output=hc4rat_test_%j_23072024_2.xt
#SBATCH --error=hc4rat_test_%j_23072024_2.err

#SBATCH -p a100
#SBATCH -N 1
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=100G  # Request 100GB of memory
#SBATCH --cpus-per-task=12  # Request 8 cores


module purge
module load miniconda/23.10.0
module load cuda/11.8
conda activate ksenvv2
nvidia-smi

python /nfs/nhome/live/carlag/neuropixels_sort/rat_robot_kilosort4_iblcbin_hc4.py
