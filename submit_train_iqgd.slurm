#!/bin/bash
#SBATCH --job-name=iqgd_training
#SBATCH --account=alkontar0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/iqgd_training_%j.out
#SBATCH --error=logs/iqgd_training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rongbo@umich.edu

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Set up environment
echo "Setting up environment..."
source /home/rongbo/env_rl.sh

# Verify environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null; then
    echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Create necessary directories
mkdir -p logs
mkdir -p logs/iqgd_training
mkdir -p models/checkpoints
mkdir -p outputs/training
mkdir -p /scratch/alkontar_root/alkontar0/rongbo/${SLURM_JOB_ID}

# Change to project directory
cd /home/rongbo/RLresearch/IQGD

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run IQGD training
echo "=========================================="
echo "Starting IQGD training..."
echo "=========================================="

python -u train_iqgd.py 2>&1 | tee logs/iqgd_training_${SLURM_JOB_ID}.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "IQGD training completed successfully!"
    echo "Models saved in models/checkpoints/"
    echo "Outputs saved in outputs/training/"
else
    echo "IQGD training failed with exit code $?"
    echo "Check logs for details"
fi

# Copy important results to scratch
echo "=========================================="
echo "Copying results..."
echo "=========================================="

cp -r models/checkpoints /scratch/alkontar_root/alkontar0/rongbo/${SLURM_JOB_ID}/ 2>/dev/null || echo "No checkpoints to copy"
cp -r outputs/training /scratch/alkontar_root/alkontar0/rongbo/${SLURM_JOB_ID}/ 2>/dev/null || echo "No outputs to copy"
cp logs/iqgd_training_${SLURM_JOB_ID}.log /scratch/alkontar_root/alkontar0/rongbo/${SLURM_JOB_ID}/ 2>/dev/null || echo "No log to copy"

echo "Results copied to /scratch/alkontar_root/alkontar0/rongbo/${SLURM_JOB_ID}/"

echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
