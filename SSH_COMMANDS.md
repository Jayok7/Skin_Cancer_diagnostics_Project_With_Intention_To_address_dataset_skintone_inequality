# SSH Command Cheatsheet

Quick reference for all SSH and remote server commands you'll need.

---

## Basic SSH Commands

### Connect to Server
```bash
ssh username@hostname
ssh m84149ji@csf3.itservices.manchester.ac.uk
```

### Connect with Specific Port
```bash
ssh -p PORT username@hostname
ssh -p 2222 m84149ji@csf3.itservices.manchester.ac.uk
```

### Disconnect
```bash
exit
# or press Ctrl+D
```

### Keep Connection Alive (Prevent Timeout)
```bash
# Add to ~/.ssh/config on LOCAL machine
Host csf3
    HostName csf3.itservices.manchester.ac.uk
    User m84149ji
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then connect with:
```bash
ssh csf3
```

---

## File Transfer Commands

### SCP (Secure Copy)

**Upload single file TO server**:
```bash
scp local_file.txt username@hostname:~/remote/path/
scp train.py m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
```

**Upload directory TO server** (recursive):
```bash
scp -r local_directory/ username@hostname:~/remote/path/
scp -r datasets/ m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/
```

**Download single file FROM server**:
```bash
scp username@hostname:~/remote/file.txt ./local/path/
scp m84149ji@csf3.itservices.manchester.ac.uk:~/outputs/model.keras ./
```

**Download directory FROM server**:
```bash
scp -r username@hostname:~/remote/directory/ ./local/path/
scp -r m84149ji@csf3.itservices.manchester.ac.uk:~/outputs/ ./
```

**Download with wildcard**:
```bash
scp username@hostname:~/outputs/*.keras ./outputs/
scp m84149ji@csf3.itservices.manchester.ac.uk:~/logs/*.log ./logs/
```

**Show progress**:
```bash
scp -v local_file.txt username@hostname:~/  # Verbose mode
```

### RSYNC (Better for Large Transfers)

**Upload directory (resumable)**:
```bash
rsync -avz --progress local_dir/ username@hostname:~/remote_dir/
rsync -avz --progress datasets/ m84149ji@csf3.itservices.manchester.ac.uk:~/skin-cancer/datasets/
```

**Download directory**:
```bash
rsync -avz --progress username@hostname:~/remote_dir/ ./local_dir/
rsync -avz --progress m84149ji@csf3.itservices.manchester.ac.uk:~/outputs/ ./outputs/
```

**Options explained**:
- `-a` = archive mode (preserves permissions, timestamps)
- `-v` = verbose (show progress)
- `-z` = compress data during transfer
- `--progress` = show per-file progress

**Exclude files**:
```bash
rsync -avz --exclude '*.tmp' --exclude 'cache/' datasets/ username@hostname:~/
```

**Dry run (test without actual transfer)**:
```bash
rsync -avzn --progress datasets/ username@hostname:~/  # -n = dry run
```

---

## Remote Execution Commands

### Execute Single Command
```bash
ssh username@hostname 'command'
ssh m84149ji@csf3.itservices.manchester.ac.uk 'ls -la ~/skin-cancer'
ssh m84149ji@csf3.itservices.manchester.ac.uk 'qstat -u m84149ji'
```

### Execute Multiple Commands
```bash
ssh username@hostname 'command1 && command2 && command3'
ssh m84149ji@csf3.itservices.manchester.ac.uk 'cd skin-cancer && ls -la && pwd'
```

### Execute Script Remotely
```bash
ssh username@hostname 'bash -s' < local_script.sh
```

---

## SSH Key Management

### Generate SSH Key Pair
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# or for older systems:
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

### Copy Public Key to Server (Password-less Login)
```bash
ssh-copy-id username@hostname
ssh-copy-id m84149ji@csf3.itservices.manchester.ac.uk
```

**Manual method** (if ssh-copy-id not available):
```bash
cat ~/.ssh/id_ed25519.pub | ssh username@hostname 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
```

### Test SSH Key Authentication
```bash
ssh -i ~/.ssh/id_ed25519 username@hostname
```

---

## Remote Directory Navigation

### List Remote Directory
```bash
ssh username@hostname 'ls -la ~/path'
ssh m84149ji@csf3.itservices.manchester.ac.uk 'ls -lh ~/skin-cancer/outputs'
```

### Check Remote Disk Space
```bash
ssh username@hostname 'df -h'
ssh m84149ji@csf3.itservices.manchester.ac.uk 'du -sh ~/skin-cancer/*'
```

### Create Remote Directory
```bash
ssh username@hostname 'mkdir -p ~/new/directory'
ssh m84149ji@csf3.itservices.manchester.ac.uk 'mkdir -p ~/skin-cancer/checkpoints'
```

### Remove Remote Files
```bash
ssh username@hostname 'rm -rf ~/path/to/file'
ssh m84149ji@csf3.itservices.manchester.ac.uk 'rm ~/skin-cancer/old_model.keras'
```

---

## Port Forwarding

### Local Port Forwarding (Access Remote Service Locally)
```bash
ssh -L local_port:localhost:remote_port username@hostname
# Example: Access remote Jupyter on local port 8888
ssh -L 8888:localhost:8888 m84149ji@csf3.itservices.manchester.ac.uk
# Then open: http://localhost:8888 in browser
```

### Remote Port Forwarding (Expose Local Service Remotely)
```bash
ssh -R remote_port:localhost:local_port username@hostname
```

### Dynamic Port Forwarding (SOCKS Proxy)
```bash
ssh -D 1080 username@hostname
```

---

## SSH Config File (~/.ssh/config)

Create `~/.ssh/config` on your LOCAL machine:

```ssh-config
Host csf3
    HostName csf3.itservices.manchester.ac.uk
    User m84149ji
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
    Compression yes

Host csf3-jump
    HostName csf3.itservices.manchester.ac.uk
    User m84149ji
    ProxyJump gateway.manchester.ac.uk
```

Then simply use:
```bash
ssh csf3
scp file.txt csf3:~/
```

---

## Common CSF-Specific Commands

### Check Job Queue
```bash
ssh m84149ji@csf3.itservices.manchester.ac.uk 'qstat -u m84149ji'
```

### Submit Job Remotely
```bash
ssh m84149ji@csf3.itservices.manchester.ac.uk 'cd ~/skin-cancer && qsub submit_job.sh'
```

### Tail Remote Log
```bash
ssh m84149ji@csf3.itservices.manchester.ac.uk 'tail -f ~/skin-cancer/logs/*.log'
```

### Check GPU Availability
```bash
ssh m84149ji@csf3.itservices.manchester.ac.uk 'qstat -f -l v100'
```

---

## Troubleshooting SSH

### Connection Timeout
```bash
# Test connection
ssh -vvv username@hostname  # Triple verbose mode

# Common fixes:
# 1. Check internet connection
# 2. Verify hostname is correct
# 3. Try different network (university VPN if off-campus)
```

### Permission Denied (Public Key)
```bash
# Check key permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# Verify key is added
ssh-add -l

# Add key if needed
ssh-add ~/.ssh/id_ed25519
```

### Host Key Verification Failed
```bash
# Remove old host key
ssh-keygen -R hostname
ssh-keygen -R csf3.itservices.manchester.ac.uk
```

### Connection Drops (Broken Pipe)
```bash
# Use screen/tmux on remote server
ssh username@hostname
screen -S training  # Create session
# Run your commands
# Press Ctrl+A then D to detach
# Reconnect later with:
screen -r training
```

---

## Screen/Tmux (Keep Sessions Alive)

### Screen Commands

**Start new session**:
```bash
screen -S session_name
screen -S csf_training
```

**Detach** (session keeps running):
```
Ctrl+A then D
```

**List sessions**:
```bash
screen -ls
```

**Reattach**:
```bash
screen -r session_name
screen -r csf_training
```

**Kill session**:
```bash
screen -X -S session_name quit
```

### Tmux Commands

**Start new session**:
```bash
tmux new -s session_name
```

**Detach**:
```
Ctrl+B then D
```

**List sessions**:
```bash
tmux ls
```

**Reattach**:
```bash
tmux attach -t session_name
```

---

## Quick Command Summary

| Task | Command |
|------|---------|
| Connect | `ssh user@host` |
| Upload file | `scp file.txt user@host:~/` |
| Download file | `scp user@host:~/file.txt ./` |
| Upload directory | `scp -r dir/ user@host:~/` |
| Download directory | `scp -r user@host:~/dir/ ./` |
| Resumable transfer | `rsync -avz --progress dir/ user@host:~/` |
| Run remote command | `ssh user@host 'command'` |
| Setup SSH key | `ssh-keygen` then `ssh-copy-id user@host` |
| Keep alive | Add `ServerAliveInterval 60` to `~/.ssh/config` |
| Port forward | `ssh -L 8888:localhost:8888 user@host` |

---

## Advanced: Multiplexing with ControlMaster

Speed up repeated connections by reusing a single connection.

Add to `~/.ssh/config`:
```ssh-config
Host *
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
```

Benefits:
- First connection: normal speed
- Subsequent connections: instant (reuses existing connection)
- Lasts 10 minutes after last use
