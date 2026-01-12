# IQGD Job Monitoring Guide

## Current Job Status

### Baseline Test Job
- **Job ID**: 39732138
- **Status**: Running (R)
- **Node**: gl1011
- **GPU**: Tesla V100-PCIE-16GB (16GB)
- **Partition**: gpu
- **Time Limit**: 1:00:00 (1 hour)
- **Started**: Mon Jan 12 07:51:29 EST 2026

---

## How to Monitor Jobs

### 1. Check Job Status
```bash
squeue -u rongbo
```
**Status codes:**
- `PD` = Pending (waiting for resources)
- `R` = Running
- `CG` = Completing
- `CD` = Completed
- `F` = Failed
- `CA` = Cancelled

### 2. View Job Output (Real-time)
```bash
# Watch the output file update
watch -n 5 "tail -50 ~/RLresearch/IQGD/logs/test_baseline_39732138.out"

# Or use tail -f for continuous monitoring
tail -f ~/RLresearch/IQGD/logs/test_baseline_39732138.out
```

### 3. Check Errors
```bash
# View error log
tail -50 ~/RLresearch/IQGD/logs/test_baseline_39732138.err

# Check if there are any errors
cat ~/RLresearch/IQGD/logs/test_baseline_39732138.err
```

### 4. View Job Details
```bash
# Detailed job information
scontrol show job 39732138

# Job accounting information (after completion)
sacct -j 39732138 --format=JobID,JobName,Elapsed,State,ExitCode
```

### 5. Cancel a Job (if needed)
```bash
scancel 39732138
```

---

## Expected Behavior

### Baseline Test Job
The job will:
1. ✓ Load environment (Python 3.11 + PyTorch 2.1.0)
2. ✓ Verify CUDA and GPU
3. ⏳ Load pretrained diffusion model (~35MB)
4. ⏳ Load fluid dataset (~20GB - this takes time!)
5. ⏳ Run diffusion sampling on 3 test batches (~10-15 min per batch)
6. ⏳ Compute metrics (MSE, PSNR, SSIM)
7. ⏳ Generate visualizations
8. ⏳ Print summary

**Total expected time**: 30-45 minutes

---

## Output Locations

### Logs
- **Standard output**: `logs/test_baseline_39732138.out`
- **Error output**: `logs/test_baseline_39732138.err`
- **Detailed log**: `logs/baseline_test_39732138.log`

### Results
- **Visualizations**: `outputs/baseline_test/batch_*.png`
- **Summary**: Printed at end of log file

---

## Common Issues and Solutions

### Issue 1: Job Pending (PD) for Long Time
```bash
# Check why job is pending
squeue -u rongbo --start

# View queue status
squeue -p gpu
```
**Solution**: Wait for resources to become available, or try a different partition

### Issue 2: Job Fails Immediately
```bash
# Check error log
cat logs/test_baseline_39732138.err
```
**Common causes**:
- Missing dataset (check `ls data/new_dataset/`)
- Missing model (check `ls models/pretrained/`)
- Memory issues (increase `--mem`)
- Import errors (check Python environment)

### Issue 3: CUDA Out of Memory
**Solution**: Reduce batch size in the script or request more GPU memory

### Issue 4: Job Timeout
**Solution**: Increase time limit in SLURM script (`--time=2:00:00`)

---

## Quick Commands Reference

```bash
# Navigate to project
cd ~/RLresearch/IQGD

# Submit job
sbatch submit_test_baseline.slurm

# Check status
squeue -u rongbo

# Monitor output
tail -f logs/test_baseline_<JOB_ID>.out

# View results after completion
ls -la outputs/baseline_test/
cat logs/test_baseline_<JOB_ID>.out | grep "BASELINE RESULTS"

# Check if job completed successfully
sacct -j <JOB_ID> --format=JobID,State,ExitCode
```

---

## After Job Completes

### 1. Check Exit Status
```bash
sacct -j 39732138 --format=JobID,State,ExitCode
```
- `ExitCode 0:0` = Success
- `ExitCode 1:0` or higher = Failed

### 2. View Final Results
```bash
# See summary
tail -100 logs/test_baseline_39732138.out | grep -A 10 "BASELINE RESULTS"

# View visualizations
ls outputs/baseline_test/

# Open images (if using VSCode with X forwarding)
code outputs/baseline_test/batch_1.png
```

### 3. Analyze Performance
```bash
# Check job runtime
sacct -j 39732138 --format=Elapsed

# Check memory usage
sacct -j 39732138 --format=MaxRSS

# Check GPU efficiency
# (requires additional setup, usually via nvidia-smi logs)
```

---

## Next Steps

Once the baseline test completes successfully:

1. **Review baseline metrics** to establish performance baseline
2. **Submit IQGD training job**:
   ```bash
   sbatch submit_train_iqgd.slurm
   ```
3. **Compare IQGD vs baseline** after training

---

## Contact & Support

- **Great Lakes HPC Support**: hpc-support@umich.edu
- **Documentation**: https://arc.umich.edu/greatlakes/
- **Project**: IQGD (Iterative Q-Guided Diffusion)

---

**Last Updated**: January 12, 2026
