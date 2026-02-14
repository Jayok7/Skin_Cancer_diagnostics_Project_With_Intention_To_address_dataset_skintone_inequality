# CSF GPU Cluster Setup Guide - Fitzpatrick Classifier

## Quick Reference Commands

```bash
# 1. SSH to CSF
ssh m84149ji@csf3.itservices.manchester.ac.uk

# 2. Submit job
qsub submit_job.sh

# 3. Check job status
qstat -u m84149ji

# 4. Download trained models
scp m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/outputs/*.keras ./
```

---

## Complete Step-by-Step Guide

### Step 1: Prepare Your Local Files

**Convert your notebook to a Python script** (already done!):
- ✅ [`train_fitzpatrick.py`](file:///d:/skin%20cancer%20project/train_fitzpatrick.py) - Training script
- ✅ [`submit_job.sh`](file:///d:/skin%20cancer%20project/submit_job.sh) - Job submission script

**Make the submission script executable**:
```powershell
# On Windows (Git Bash or WSL)
chmod +x submit_job.sh
```

---

### Step 2: Connect to CSF via SSH

**Initial connection**:
```bash
ssh m84149ji@csf3.itservices.manchester.ac.uk
# Replace 'username' with your CSF username
```

**First-time setup** (password + 2FA):
- Enter your password
- Enter your 2FA code from authenticator app

**Optional: Setup SSH key for easier access**:
```bash
# On your local machine (Git Bash/PowerShell)
ssh-keygen -t ed25519 -C "your.email@domain.com"
ssh-copy-id m84149ji@csf3.itservices.manchester.ac.uk
```

---

### Step 3: Setup Your Environment on CSF

**Create project directory**:
```bash
cd ~
mkdir -p skin-cancer
cd skin-cancer
mkdir -p datasets logs outputs
```

**Load modules**:
```bash
# Check available modules
module avail

# Load Anaconda and CUDA
module load apps/binapps/anaconda3/2021.11
module load libs/cuda/11.8.0
```

**Create conda environment**:
```bash
# Create environment with TensorFlow GPU
conda create -n tf_gpu python=3.10 -y
source activate tf_gpu

# Install required packages
pip install tensorflow[and-cuda]
pip install pandas numpy matplotlib seaborn scikit-learn tqdm
pip install pillow
```

**Verify GPU access**:
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

### Step 4: Transfer Your Data and Code

**Option A: Using SCP (Simple)**
```bash
# From your LOCAL machine (Git Bash/PowerShell)
cd "d:/skin cancer project"

# Transfer training script
scp train_fitzpatrick.py m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/

# Transfer job script
scp submit_job.sh m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/

# Transfer dataset (this may take a while!)
scp -r datasets m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
```

**Option B: Using Git (Recommended if you have GitHub)**
```bash
# On CSF
cd ~/skin-cancer
git clone https://github.com/jayok7/skin-cancer-classifier.git .

# If datasets are not in Git (too large), use SCP for data only
```

**Option C: Using rsync (Fast, resume-able)**
```bash
# From local machine
rsync -avz --progress "d:/skin cancer project/datasets" m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
rsync -avz --progress "d:/skin cancer project/train_fitzpatrick.py" m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
rsync -avz --progress "d:/skin cancer project/submit_job.sh" m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
```

---

### Step 5: Configure the Job Script

**Edit the submission script** on CSF:
```bash
cd ~/skin-cancer
nano submit_job.sh
```

**Update these lines**:
```bash
#$ -M your.email@domain.com  # CHANGE to your email

# Update paths if different
source activate tf_gpu  # Your conda environment name

# Adjust GPU type if needed:
#$ -l v100    # Options: v100, a100, t4 (check with: qconf -sc)

# Adjust memory if needed:
#$ -l mem256  # Options: mem64, mem128, mem256
```

---

### Step 6: Submit Your Training Job

**Submit the job**:
```bash
cd ~/skin-cancer
qsub submit_job.sh
```

**Expected output**:
```
Your job 1234567 ("fitzpatrick_train") has been submitted
```

**Check job status**:
```bash
# View your jobs
qstat -u m84149ji

# View detailed job info
qstat -j 1234567

# View all GPU jobs
qstat -l v100
```

**Job status codes**:
- `qw` = queued/waiting
- `r` = running
- `t` = transferring
- `Eqw` = error (check logs)

---

### Step 7: Monitor Training Progress

**View real-time logs**:
```bash
# Watch log file update
tail -f logs/fitzpatrick_1234567.log

# Exit with Ctrl+C
```

**Check GPU usage** (if job is running):
```bash
# SSH to the compute node (find node name in qstat output)
qrsh -l v100
nvidia-smi

# Or check remotely
ssh compute-node-name nvidia-smi
```

**Expected training time**:
- 3-way classification: ~2-4 hours on V100
- 6-way classification: ~3-5 hours on V100

---

### Step 8: Retrieve Trained Models

**After job completes**, download models to your local machine:

```bash
# From your LOCAL machine (Git Bash/PowerShell)
cd "d:/skin cancer project"

# Download all model files
scp m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/outputs/*.keras ./outputs/

# Download specific model
scp m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/outputs/best_finetuned_model.keras ./

# Download logs
scp m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/logs/*.log ./logs/
```

---

## Troubleshooting

### Job fails immediately (Eqw status)
```bash
# Check error
qstat -j 1234567 | grep error

# Common fixes:
# 1. Check paths in submit_job.sh
# 2. Verify conda environment exists
# 3. Check file permissions: chmod +x submit_job.sh
```

### Out of memory error
```bash
# Increase memory in submit_job.sh
#$ -l mem512  # Request more memory

# Or reduce batch size in train_fitzpatrick.py
--batch-size 4  # Reduce from 16
```

### CUDA out of memory
```bash
# Reduce batch size or image size
--batch-size 4
--image-size 224
```

### Can't find GPU
```bash
# Check CUDA module loaded
module list

# Reload CUDA
module load libs/cuda/11.8.0

# Verify TensorFlow sees GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Advanced: Interactive GPU Session

For **testing/debugging** (not training):
```bash
# Request interactive GPU session
qrsh -l v100 -pe smp.pe 4

# Once on GPU node
source activate tf_gpu
python

# Test imports
>>> import tensorflow as tf
>>> tf.config.list_physical_devices('GPU')
>>> exit()

# Exit session
exit
```

---

## File Structure on CSF

After setup, your CSF directory should look like:
```
~/skin-cancer/
├── train_fitzpatrick.py         # Training script
├── submit_job.sh                 # Job submission script
├── datasets/
│   ├── MSKCC-images/             # Image files
│   └── mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv
├── outputs/                      # Created by script
│   ├── best_head_model.keras
│   ├── best_finetuned_model.keras
│   └── final_model.keras
└── logs/                         # Job logs
    └── fitzpatrick_1234567.log
```

---

## Expected Output Files

After successful training, you'll get:

1. **`best_head_model.keras`** - Best model from Stage 1 (head training)
2. **`best_finetuned_model.keras`** - Best model from Stage 2 (fine-tuning) ⭐ USE THIS
3. **`final_model.keras`** - Model at end of training
4. **`fitzpatrick_JOBID.log`** - Complete training log with accuracy metrics

---

## Next Steps After Downloading Models

Once you have the trained models locally:

1. **Load in Jupyter**:
```python
from tensorflow.keras.models import load_model
model = load_model('outputs/best_finetuned_model.keras', 
                   custom_objects={'loss_fn': focal_loss()})
```

2. **Run evaluation locally** using Section 7 enhanced evaluation
3. **Deploy** to your application/API
