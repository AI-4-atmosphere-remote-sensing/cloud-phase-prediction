## AWS Automation Using Multi-GPU support
This guide explains how to automate running the ```Cloud Phse Prediction``` Model on AWS EC2 instances using Multi GPU ```(please note: this approach only supports single node multi-gpu as of now.)```. The ```aws-automation-cloud_pahse_prediction.py``` script automates launching an EC2 instance, executing ML Model, and storing results in S3 and desired location in locally, finally terminating the initiated EC2 instance..

## Prerequisites

### 1. AWS Account Setup
1. Create an AWS account if you don't have one
2. Create IAM user with following permissions:
   - EC2 full access
   - S3 full access
   - Get your Access Key ID and Secret Access Key from IAM

### 2. Local Termina (Environment) Setup
1. Install Python requirements:
   ```bash
   pip install awscli boto3 paramiko
   ```

2. Configure AWS CLI:
   ```bash
   aws configure
   ```
   Enter:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (use us-west-2)
   - Default output format (json)

3. Create S3 bucket:
   - Create a bucket in us-west-2 region
   - Make sure bucket has public access enabled
   - Note the bucket name for script parameters

4. EC2 Key Pair:
   - Create or use existing EC2 key pair in us-west-2 region
   - Save the .pem file locally
   - Set proper permissions: `chmod 400 your-key.pem`

### 3. Code Preparation
1. Prepare your code directory structure:
   ```
   your-code/
   ├── main.py           # Main ML script
   └── [other code files]
   ```

2. Zip your code and data files:
   ```bash
   zip -r code_and_data.zip your-code/
   ```
## Running the Script

### Basic Usage
```bash
python3 aws-automation-cloud_pahse_prediction.py \
  --key_path /path/to/your-key.pem \
  --code_zip /path/to/code.zip \
  --s3_bucket your-bucket-name \
  --output_dir /path/to/loacl/output/dir \
  --epochs 4 \
  --num_of_gpu 4 \
```

### Required Parameters
- `--key_path`: Path to EC2 key pair file (.pem)
- `--code_zip`: Path to zipped code files
- `--output_dir`: local dir destination for storing results
- `--s3_bucket`: S3 bucket name for storing results

## Workflow (how the script works)

### 1. Instance Launch
- Launches g5.12xlarge instance with GPU support
- Uses AMI: ami-0339ea6f7f5408bb9 (includes CUDA and ML libraries)
- Automatically configures security group access

### 2. Code Deployment
- Uploads and extracts your code and data
- Installs requirements:
  ```
  torchinfo
  mmcv==1.6.2
  h5py
  scikit-image
  ```
- Special installation for mmcv with CUDA support

### 3. Execution
- Locates and runs main.py with specified parameters
- Streams real-time output to your terminal
- Captures all output

### 4. Results Storage
Results are stored in S3 with timestamp:
```
s3://your-bucket/result_code_YYYYMMDD_HHMMSS/
├── log.txt  # Complete execution log including setup
└── out.txt  # ML model output only
```

### 5. Cleanup
- Automatically terminates EC2 instance
- All results preserved in S3

## Troubleshooting
- Ensure AWS credentials are configured correctly
- Check S3 bucket permissions
- Verify .pem file permissions (chmod 400)
- Make sure code.zip contains main.py at the correct path
- Check EC2 instance limits in your AWS account

## Cost Management
- Script uses g4dn.12xlarge instance (costly)
- Instance terminates automatically after completion
- Monitor AWS billing dashboard
- Set up billing alerts

## Security Notes
- Keep .pem file secure
- Don't commit AWS credentials
- Use IAM roles with minimum required permissions
- Monitor AWS CloudTrail for security events
