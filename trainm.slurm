#!/bin/bash
#SBATCH --job-name=2nodesdata11
#SBATCH --output=n4slurm.out
#SBATCH --error=n4slurm.err
#SBATCH --partition=gpu2018
#SBATCH --qos=high_mem
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

module load Python/3.7.6-intel-2019a
nvidia-smi
srun python train.py --training_data_path='./example/training_data/'  --model_saving_path='./saved_model/'
