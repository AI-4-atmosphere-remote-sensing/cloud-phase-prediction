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
