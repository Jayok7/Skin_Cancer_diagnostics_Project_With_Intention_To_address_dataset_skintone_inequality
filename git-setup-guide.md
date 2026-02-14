# Git Setup Guide for Skin Cancer Project

## Current Status
✅ Git repository initialized  
✅ `.gitignore` file created  

## Your Git Configuration
- Git is installed at: `D:\git\Git\bin\git.exe`
- Git is NOT in your system PATH, so you need to use the full path

## Commands You'll Need

### Option 1: Use Full Path (Works Now)
```powershell
& "D:\git\Git\bin\git.exe" [command]
```

### Option 2: Add Git to PATH Permanently
To avoid typing the full path every time:

1. Press `Win + X` → Select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System variables", find and select "Path"
5. Click "Edit" → "New"
6. Add: `D:\git\Git\bin`
7. Click "OK" on all windows
8. **Restart PowerShell**

## Setup Steps

### 1. Configure Git (First Time Only)
```powershell
# Set your name (replace with your actual name)
& "D:\git\Git\bin\git.exe" config user.name "Your Name"

# Set your email (replace with your GitHub email)
& "D:\git\Git\bin\git.exe" config user.email "your.email@example.com"
```

### 2. Make Your First Commit
```powershell
# Add all files (respects .gitignore)
& "D:\git\Git\bin\git.exe" add .

# Create initial commit
& "D:\git\Git\bin\git.exe" commit -m "Initial commit: Skin cancer classification project"
```

### 3. Create GitHub Repository
1. Go to https://github.com and sign in
2. Click "+" → "New repository"
3. Name it: `skin-cancer-classifier`
4. Choose Public or Private
5. **Don't** initialize with README
6. Click "Create repository"

### 4. Connect to GitHub
```powershell
# Add remote (replace YOUR_USERNAME)
& "D:\git\Git\bin\git.exe" remote add origin https://github.com/YOUR_USERNAME/skin-cancer-classifier.git

# Set branch name to main
& "D:\git\Git\bin\git.exe" branch -M main

# Push to GitHub
& "D:\git\Git\bin\git.exe" push -u origin main
```

### 5. Future Updates
```powershell
# Check what changed
& "D:\git\Git\bin\git.exe" status

# Add changes
& "D:\git\Git\bin\git.exe" add .

# Commit changes
& "D:\git\Git\bin\git.exe" commit -m "Description of changes"

# Push to GitHub
& "D:\git\Git\bin\git.exe" push
```

## What's Being Tracked?

The `.gitignore` file excludes:
- ✅ Virtual environment (`.venv/`)
- ✅ IDE settings (`.idea/`)
- ✅ Data files (`data/` folder)
- ✅ Large model files
- ✅ Python cache files

## Files That WILL Be Tracked:
- All Python scripts (`.py` files)
- Jupyter notebooks (`.ipynb` files)
- `README.md`
- `requirements.txt`
- Project structure

## Quick Reference Card
Create an alias by adding this to your PowerShell profile:
```powershell
function git { & "D:\git\Git\bin\git.exe" $args }
```

To edit your PowerShell profile:
```powershell
notepad $PROFILE
```

After adding the alias, restart PowerShell and you can use `git` normally!
