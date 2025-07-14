## CHIP based Execution using Multi-GPU support
This guide explains how to automate running the ```Cloud Phse Prediction``` Model on CHIP cluster using Multi GPU ```(please note: this approach only supports single node multi-gpu as of now.)```. The ```chip_cloud_pahse_prediction.py``` script allow running the code in CHIP cluster.

## Prerequisites

### 1. CHIP Account Setup
1. Request a CHIP account if you don't have one yet. [UMBC HPCF](https://hpcf.umbc.edu/)
2. Make sure you have all the permissions to setup enviromnents (like, conda environment).

### 2. Environment Setup
1. Login to CHIP using UMBC credectials.
2. Go to your right directory
3. Make sure the Conda Environment (or Python environment) is ready with necessary libraries (look for ```requirements.txt``` bellow).

### 3. Upload/Download all necessary files and code
1. You can either use ```git clone repo``` or you can upload your all files/data using ```scp -r your/zip/file user@chip.rs.umbc.edu:user/your/path``` if they are in you local machine.
2. Make sure you have all necessary files, such as the main file (```chip_cloud_pahse_prediction.py```) to run and all data paths are correctly mentioned in the code file.
3. Following is the ```SLURM``` file to be used to run the batch jobs. You can update as necessary.

```
#!/bin/bash
#SBATCH --job-name=time                     # Job name
#SBATCH --cluster=chip-gpu		              # specify gpu or cpu
#SBATCH --mem=20GB                            # Job memory request
#SBATCH --gres=gpu:2                          # Number of requested GPU(s) ## you can choose up to the max GPU number.
#SBATCH --time=72:00:00                       # Time limit days-hrs:min:sec
##SBATCH --constraint=L40S #H100	            # Specific hardware constraint ## commenting this line out will choose one randomly from whatever is available
#SBATCH --error=err.err                       # Error file name
#SBATCH --output=out.out                      # Output file name

module load Anaconda3
echo "Activating conda"
eval "$(conda shell.bash activate /umbc/rs/nasa-access/your_env)" ## this line is required in chip, this line will create a conda.sh in your chip home dir automatically. Here, your_env is the name of your environment, like, my-conda-env.
echo $CONDA_PREFIX
#eval "(conda shell.bash hook)"
#conda activate /umbc/rs/nasa-access/your_env
srun python chip_cloud_pahse_prediction.py --training_data_path='./example/training_data/'  --model_saving_path='./saved_model/'
```

### 4. Training Completion
1. The training will create all necessary output as it is designed, as well as it will create ```out.out``` and ```err.err``` for any output and errors. So please have a look at those files.
2. Once the training is completed, make sure to logout from CHIP cluster.
