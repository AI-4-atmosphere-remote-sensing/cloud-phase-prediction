#!/bin/bash
#SBATCH --job-name=accesstrainmh2016_gpu2_M8M16
#SBATCH --account=cybertrn
#SBATCH --partition=gpu2018
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --qos=medium+
#SBATCH --mem=8000
#SBATCH --time=24:00:00
#SBATCH --output=./slurm_log/slurm-%x-%j-%u.out
#SBATCH --error=./slurm_log/slurm-%x-%j-%u.err

srun python train.py --training_data_path='/home/xinh1/nasa-access/xin/cloud-phase-prediction/data/training_2016/'  --model_saving_path='/home/xinh1/nasa-access/xin/cloud-phase-prediction/saved_model_2016_M8_M16_polarROnly'
