#!/bin/bash
#SBATCH --job-name=ToFindPolar_accesspredmh2017M8M16_001_gpu2
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

srun python predict_chenxi_onlyM16_granule-bak_mhumair1V4_M8M16_onlyPolarR.py --predicting_data_path '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/pct_2016_M8M16/onlyPolarReg/2017/001/' --model_saving_path '/umbc/rs/nasa-access/xin/cloud-phase-prediction/saved_model_2016_M8_M16_polarROnly/' --export_data_path '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/predict_2016_M8M16PolarReg/2017/001/'
