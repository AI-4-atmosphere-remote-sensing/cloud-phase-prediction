#!/bin/bash
#SBATCH --job-name=accessJobCollocation
#SBATCH --account=cybertrn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --qos=long+
#SBATCH --mem=MaxMemPerNode
#SBATCH --time=11-24:00:00
#SBATCH --output=./slurm_log/slurm-%x-%j-%u.out
#SBATCH --error=./slurm_log/slurm-%x-%j-%u.err

module load Python/3.7.6-intel-2019a
python ./evaluate.py --testing_data_path='/umbc/rs/nasa-access/data/labels4/npz/test/'  --model_saving_path='/umbc/rs/nasa-access/data/labels4/npz/saved_model/gpu/'
