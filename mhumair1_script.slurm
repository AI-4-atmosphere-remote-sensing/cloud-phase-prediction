#!/bin/bash
#SBATCH --job-name=accesstrain2016mh
#SBATCH --account=cybertrn
#SBATCH --partition=high_mem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --qos=long+
#SBATCH --mem=8000
#SBATCH --time=11-24:00:00
#SBATCH --output=./slurm_log/slurm-%x-%j-%u.out
#SBATCH --error=./slurm_log/slurm-%x-%j-%u.err

module load Python/3.7.6-intel-2019a
srun python train.py --training_data_path='/home/xinh1/nasa-access/xin/cloud-phase-prediction/data/training_2016/'  --model_saving_path='/home/xinh1/nasa-access/xin/cloud-phase-prediction/saved_model_2016'

