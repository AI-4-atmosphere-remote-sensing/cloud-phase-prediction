#!/bin/bash
#SBATCH --job-name=accessJobCollocation
#SBATCH --account=cybertrn
#SBATCH --partition=gpu2018
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --qos=medium+
#SBATCH --mem=MaxMemPerNode
#SBATCH --time=24:00:00
#SBATCH --output=/home/xinh1/access/ieee/slurm_log/slurm-%x-%j-%u.out
#SBATCH --error=//home/xinh1/access/ieee/slurm_log/slurm-%x-%j-%u.err

module load Python/3.7.6-intel-2019a
python totaki_4yr_train.py
