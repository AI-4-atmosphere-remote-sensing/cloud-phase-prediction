#!/bin/bash
#SBATCH --job-name=accessTrain2016
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

module load Python/3.7.6-intel-2019a
python train.py --training_data_path='/home/xinh1/nasa-access/xin/data/training_2016/'  --model_saving_path='/home/xinh1/nasa-access/xin/data/saved_model_2016/'
