#!/bin/bash
#SBATCH --job-name=accesspredmh2017_031_gpu2
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

srun python predict_chenxi_onlyM16_granule-bak_mhumair1V3.py --predicting_data_path='/home/xinh1/nasa-access/xin/cloud-phase-prediction/data/predict_2016/2017/031/'  --model_saving_path='/home/xinh1/nasa-access/xin/cloud-phase-prediction/saved_model_2016/' --export_data_path='/home/xinh1/nasa-access/xin/cloud-phase-prediction/data/predict_2016/2017/031/'
