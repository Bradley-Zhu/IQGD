#!/bin/bash
#SBATCH --job-name=iqgd_test_baseline
#SBATCH --account=alkontar0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/test_baseline_%j.out
#SBATCH --error=logs/test_baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rongbo@umich.edu

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
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
mkdir -p outputs/baseline_test

# Change to project directory
cd /home/rongbo/RLresearch/IQGD

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run baseline test
echo "=========================================="
echo "Testing MGDM baseline diffusion model..."
echo "=========================================="

python -u test_baseline.py 2>&1 | tee logs/baseline_test_${SLURM_JOB_ID}.log

# Check if test completed successfully
if [ $? -eq 0 ]; then
    echo "Baseline test completed successfully!"
    echo "Results saved in outputs/baseline_test/"
else
    echo "Baseline test failed with exit code $?"
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
