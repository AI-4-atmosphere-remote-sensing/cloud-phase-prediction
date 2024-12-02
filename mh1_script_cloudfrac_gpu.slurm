#!/bin/bash
#SBATCH --job-name=accessCloudFraction_mh2017_gpu2GT
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

srun python cal_cloudFraction_7daysGT.py
