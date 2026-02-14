# Quick Start: Running on CSF GPU Cluster

## 🚀 The Fastest Path

### On Your Local Machine:
```bash
cd "d:/skin cancer project"

# 1. Transfer files to CSF
scp train_fitzpatrick.py m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
scp submit_job.sh m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
scp -r datasets m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
```

### On CSF (SSH):
```bash
# 1. Connect
ssh m84149ji@csf3.itservices.manchester.ac.uk

# 2. Setup environment (FIRST TIME ONLY)
cd ~/skin-cancer
module load apps/binapps/anaconda3/2021.11
module load libs/cuda/11.8.0
conda create -n tf_gpu python=3.10 -y
source activate tf_gpu
pip install tensorflow[and-cuda] pandas numpy matplotlib seaborn scikit-learn tqdm pillow

# 3. Edit email in submit script
nano submit_job.sh  # Change email address

# 4. Submit job
qsub submit_job.sh

# 5. Monitor
qstat -u m84149ji
tail -f logs/fitzpatrick_*.log
```

### Download Models (After Training):
```bash
# From local machine
scp m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/outputs/*.keras ./outputs/
```

---

## 📋 Command Cheat Sheet

| Task | Command |
|------|---------|
| Connect to CSF | `ssh m84149ji@csf3.itservices.manchester.ac.uk` |
| Submit job | `qsub submit_job.sh` |
| Check status | `qstat -u m84149ji` |
| View logs | `tail -f logs/fitzpatrick_*.log` |
| Cancel job | `qdel JOBID` |
| Download models | `scp m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/outputs/*.keras ./` |

---

## ⚙️ Customize Training

Edit the **last part of `submit_job.sh`**:

```bash
python train_fitzpatrick.py \
    --use-3way              # For 3-way classification
    # --use-6way            # Remove --use-3way for 6-way
    --batch-size 16         # Increase if you have more GPU memory
    --epochs-head 15        # Head training epochs
    --epochs-finetune 60    # Fine-tuning epochs
    --image-size 260        # Image resolution
```

---

## 📚 Full Documentation

See [CSF_GPU_GUIDE.md](file:///d:/skin%20cancer%20project/CSF_GPU_GUIDE.md) for:
- Detailed environment setup
- Troubleshooting
- Advanced usage
- Interactive debugging sessions

---

## ⏱️ Expected Timeline

| Phase | Time |
|-------|------|
| Data transfer | 10-30 min |
| Queue wait | 5 min - 2 hours |
| Training (3-way) | 2-4 hours |
| Training (6-way) | 3-5 hours |

Total: **3-8 hours** from start to finished models
