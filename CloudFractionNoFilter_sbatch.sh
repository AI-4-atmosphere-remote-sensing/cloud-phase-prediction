#!/bin/bash
#SBATCH --job-name=CloudFractionNoFilter_nasaAccess
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_log/slurm-%x-%j-%u.out
#SBATCH --error=./slurm_log/slurm-%x-%j-%u.err

module purge
module load Python/3.11.3-GCCcore-12.3.0
source /umbc/rs/nasa-access/users/mhumair1/venv_cloud/bin/activate

python -u cal_cloudFraction_31daysNOFilterSameScale.py

