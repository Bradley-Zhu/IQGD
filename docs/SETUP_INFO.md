# IQGD Project Setup Information

## Project Created: 2026-01-11

### Directory Structure
```
~/RLresearch/IQGD/
├── configs/          # Configuration files for experiments
├── experiments/      # Experiment scripts and notebooks
├── logs/            # Training logs and metrics
├── models/          # Saved model checkpoints
├── outputs/         # Generated outputs and visualizations
├── utils/           # Utility functions and helpers
├── .git/            # Git repository
├── .gitignore       # Git ignore rules
├── __init__.py      # Package initialization
├── README.md        # Project documentation
└── requirements.txt # Python dependencies
```

## Environment Information

### System
- **OS**: Linux (RHEL 8)
- **Git Version**: 2.43.7
- **Python Version**: 3.10.9 (base environment)
- **Conda**: Installed at ~/anaconda3

### Available Conda Environments
- `base` (active by default) - Python 3.10.9
- `RL` - Existing RL environment
- `dppo` - DPPO project environment
- `mae` - MAE project environment

### Available HPC Modules
- **PyTorch**: pytorch/1.12.1, pytorch/2.0.1
- **JAX**: jax/0.4.30
- **Mamba**: mamba/py3.10, py3.11, py3.12, py3.13
- **Matplotlib**: matplotlib/3.5.3
- And many more...

## Git Repository

### Status
- Repository initialized ✓
- Files staged for initial commit
- Ready to commit with your credentials

### Important Git Configuration Needed
Before making your first commit, you need to configure git with your credentials:

```bash
# Set your git username and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Or set locally for this project only
cd ~/RLresearch/IQGD
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Making Your First Commit
```bash
cd ~/RLresearch/IQGD
git commit -m "Initial IQGD project setup"
```

### Connecting to Remote Repository
After creating a repository on GitHub/GitLab:
```bash
# Add remote
git remote add origin <your-repository-url>

# Push to remote
git branch -M main  # Rename to main if preferred
git push -u origin main
```

## Next Steps

### 1. Configure Git (Required)
Set your git username and email as shown above.

### 2. Create Conda Environment (Recommended)
```bash
# Create dedicated environment for IQGD
conda create -n iqgd python=3.10
conda activate iqgd

# Install dependencies
cd ~/RLresearch/IQGD
pip install -r requirements.txt
```

### 3. Or Use HPC Modules (Alternative)
```bash
# Load PyTorch module
module load pytorch/2.0.1

# Install additional packages
pip install --user -r requirements.txt
```

### 4. Start Development
- Add your agent implementation in `iqgd_agent.py`
- Create training script in `train.py`
- Add evaluation script in `eval.py`
- Store configurations in `configs/`

### 5. Version Control
```bash
# Make initial commit
git commit -m "Initial IQGD project setup"

# Create remote repository and push
git remote add origin <your-repo-url>
git push -u origin master
```

## Helpful Commands

### Environment Management
```bash
# List conda environments
conda env list

# Activate IQGD environment
conda activate iqgd

# Check installed packages
conda list
```

### Module Management
```bash
# Search for modules
module keyword <keyword>

# Load a module
module load <module-name>

# List loaded modules
module list
```

### Git Workflow
```bash
# Check status
git status

# Stage changes
git add <files>

# Commit changes
git commit -m "message"

# Push to remote
git push

# Pull from remote
git pull
```

## Resources
- Project README: [README.md](README.md)
- Dependencies: [requirements.txt](requirements.txt)
- Git ignore rules: [.gitignore](.gitignore)
