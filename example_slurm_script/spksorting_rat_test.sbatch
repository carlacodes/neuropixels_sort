#!/bin/bash
#SBATCH --job-name=rat_test_09042024_30
#SBATCH --output=rat_test_%j_09042024_30.txt
#SBATCH --error=rat_test_%j_09042024_30.err

#SBATCH -p a100
#SBATCH -N 1
#SBATCH --time=0-12:00
#SBATCH --gres=gpu:a100:1
#SBATCH --gres-flags=enforce-binding

module purge
module load miniconda/23.10.0
module load cuda/11.8
conda activate ksenv4
nvidia-smi

python /nfs/nhome/live/carlag/neuropixels_sort/rat_robot_kilosort4.py
