## Notices:

“Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.   All Rights Reserved.”


### Disclaimer:

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

## PyTorch Distributed Data Parallel Implementation for Cloud Phase Prediction Application

The project currently supports python>=3.7
# Step 1: Taki configuration:
1. The user must be having a taki account
2. The user must be able to access the account with their credentials.
3. Open the Windows PowerShell, and access your account as given example below: ssh garimak1@taki.rs.umbc.edu
4. Enter your password
5. You should be able to log in to the taki cluster successfully.

# Step 2: Conda Environment Set Up:
1. Firstly set up a directory to install conda.
2. Use this URL to install the conda: https://docs.conda.io/en/latest/miniconda.html
3. Use the below command to execute the .sh file:- sh Miniconda3-latest-Windows- x86 64.sh
4. Perform the necessary steps to install all the packages required for building up the base python environment
5. Check that conda is installed successfully.
6. Now try to create a conda environment using the command: 
   conda create -n cloud-phase-prediction-env -c conda-forge python=3.7 pytorch h5py pyhdf
7. Here cloud-phase-prediction-env is a random conda environment you can specify any name as per your requirement. This command will help  to set up your python virtual environment for 3.7
8. After completing this step we need to activate the newly created conda environment, we can do this by this command: conda activate   cloud-phase-prediction-env

# Step 3: Clone the Source Code: 
1. Now all the required prerequisite steps are completed and we will be shifting toward the source code. So, in the selected directory we need to clone the Project cloud-phase-prediction.
2. Clone Cloud Phase Prediction project: (https://github.com/AI-4-atmosphere-remote-sensing/cloud-phase-prediction.git)
3. Create a directory to place the cloned project and go to the directory for eg: cd cloud-phase-prediction 

# Step 4: Data Preprocessing You can skip this part and use the already preprocessed data available in the example folder.

# Step 5: To execute the slurm file use the following command: sbatch train_ddp.slurm
Creating a slurm file: A Slurm file is a script used to organize and run distributed training jobs over numerous nodes in a high-performance computing (HPC) cluster when using PyTorch DDP (Distributed Data Parallel) training.<br />
Below is the slurm file used for cloud-phase-prediction project: <br /><br />
#!/bin/bash <br />
#SBATCH --job-name=2nodesdata11 <br />
#SBATCH --output=n4slurm.out <br />
#SBATCH --error=n4slurm.err <br />
#SBATCH --partition=gpu2018 <br />
#SBATCH --qos=high_mem <br />
#SBATCH --time=01:30:00 <br />
#SBATCH --gres=gpu:4 <br />
#SBATCH --nodes=1 <br />
#SBATCH --tasks-per-node=1 <br />
module load Python/3.7.6-intel-2019a <br />
nvidia-smi <br />
srun python train_ddp.py --training_data_path='./example/training_data/' --model_saving_path='./saved_model/' <br />

Note: The updated code changes are made into train_ddp.py and data_utils_ddp.py

# Step 6: Observe the results:
  a. Check the train_ddp.slurm <br />
  b. Check the job name <br />
  c. Slurm.out file name <br />
  d. Slurm.err file name <br />

# Step 7: For the output open the slurm.out file use the following command:
 more slurm.out
