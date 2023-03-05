# Cloud Phase Prediction with Multi-Sensor Domain Adaptation 
This repository provides an end-to-end deep domain adaptation with domain mapping and correlation alignment (DAMA) and apply it to classify the heterogeneous remote satellite cloud and aerosol types.

## Prerequisite
The project currently supports `python>=3.7`

Step 1: Taki configuration:

1. The user must be having a taki account

2. The user must be able to access the account with their credentials.

3. Open the Windows PowerShell, and access your account as given example below:
ssh garimak1@taki.rs.umbc.edu

4. Enter your password

5. You should be able to log in to the taki cluster successfully.

Step 2: Conda Environment Set Up:

1. Firstly set up a directory to install conda.

2. Use this URL to install the conda: https://docs.conda.io/en/latest/miniconda.html

3. Use the below command to execute the .sh file:- sh Miniconda3-latest-Windows-
x86 64.sh

4. Perform the necessary steps to install all the packages required for building up
the base python environment

5. Check that conda is installed successfully.

6. Now try to create a conda environment using the command:
conda create -n cloud-phase-prediction-env -c conda-forge python=3.7 pytorch h5py pyhdf

Here cloud-phase-prediction-env is a random conda environment you can specify any name as per your requirement.This command will help to set up your python virtual environment
for 3.7

7. After completing this step we need to activate the newly created conda envi-
ronment, we can do this by this command:
conda activate cloud-phase-prediction-env

Step 3: Clone the Source Code: 
Now all the required prerequisite steps
are completed and we will be shifting toward the source code. So, in the selected
directory we need to clone the Project COT retrieval.

1. Change to the project directory: 
For eg: cd /umbc/rs/nasa-access/users/garimak1/ddp/cloud-phase-prediction-main

2. COT retrieval source code from the big-data-lab-umbc repository using git (cloud_phase_ddp) branch:
https://github.com/AI-4-atmosphere-remote-sensing/cloud-phase-prediction.git

3. Go to the directory:
cd cloud-phase-prediction
pip install

Step 4: Data Preprocessing
You can skip this part and use the already preprocessed data available in the example folder.

Step 5: To execute the slurm file use the following command:
sbatch trainm.slurm

Creating a slurm file: A Slurm file is a script used to organize and
run distributed training jobs over numerous nodes in a high-performance computing
(HPC) cluster when using PyTorch DDP (Distributed Data Parallel) training.

Below is the slurm file used for COT retrieval project:
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

Step 6: Observe the results:
a. Check the trainm.slurm 
b. Check the job name 
c. Slurm.out file name 
d. Slurm.err file name 

Step 7: For the output open the slurm.out file use the following command
 more slurm.out

