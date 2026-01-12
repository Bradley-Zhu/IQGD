# IQGD: Iterative Q-Guided Diffusion

## Overview
Iterative Q-Guided Diffusion (IQGD) is a reinforcement learning research project that combines Q-learning with diffusion models for improved guidance in fluid dynamics simulations.

**Based on**: MGDM (Multi-Fidelity Guided Diffusion Models)

## Key Idea
Instead of using fixed gradient-based physics guidance, IQGD uses a **learned Q-function** to guide the diffusion sampling process, enabling adaptive and optimized guidance strategies.

## Project Structure
```
IQGD/
├── docs/                # 📚 Documentation
│   ├── README.md        # Documentation index
│   ├── SETUP_INFO.md
│   ├── PHASE1_COMPLETE.md
│   ├── JOB_MONITORING.md
│   └── BASELINE_TEST_RESULTS.md
├── iqgd/               # Core IQGD modules
│   ├── data_loader.py
│   ├── diffusion_env.py
│   ├── q_network.py
│   ├── iqgd_agent.py
│   └── replay_buffer.py
├── configs/            # Configuration files
├── experiments/        # Experiment notebooks
├── logs/               # Training and job logs
├── models/             # Model checkpoints
│   └── pretrained/     # Pretrained diffusion models
├── outputs/            # Generated outputs
├── data/               # → MGDM fluid dataset (symlink)
├── test_baseline.py    # Test baseline model
├── train_iqgd.py       # Train IQGD agent
└── requirements.txt    # Dependencies
```

## Quick Start

### 1. Environment Setup
```bash
# On Great Lakes HPC
source /home/rongbo/env_rl.sh
```

### 2. Test Baseline Model
```bash
cd ~/RLresearch/IQGD
sbatch submit_test_baseline.slurm
```

### 3. Train IQGD Agent
```bash
sbatch submit_train_iqgd.slurm
```

## Documentation

📖 **See [docs/](docs/) for detailed documentation:**

- **[docs/SETUP_INFO.md](docs/SETUP_INFO.md)** - Environment and setup guide
- **[docs/PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md)** - Phase 1 implementation summary
- **[docs/JOB_MONITORING.md](docs/JOB_MONITORING.md)** - How to monitor SLURM jobs
- **[docs/BASELINE_TEST_RESULTS.md](docs/BASELINE_TEST_RESULTS.md)** - Baseline performance results

## Current Status

✅ **Phase 1 Complete** (January 11, 2026)
- Project setup and core modules implemented
- Baseline tested: **PSNR 36.98 dB**, SSIM 1.0000

🎯 **Phase 2: In Progress**
- Training IQGD agent
- Target: PSNR ≥ 37 dB

## Results

### Baseline Performance
- **PSNR**: 36.98 dB
- **SSIM**: 1.0000
- **MSE**: 0.000472

See [docs/BASELINE_TEST_RESULTS.md](docs/BASELINE_TEST_RESULTS.md) for details.

## Repository

**GitHub**: https://github.com/Bradley-Zhu/IQGD

## Contact

- **Author**: Bradley-Zhu (rongbo@umich.edu)
- **Institution**: University of Michigan
- **HPC**: Great Lakes
