# Phase 1: Project Setup & Data Preparation - COMPLETE ✓

## Date: 2026-01-11

---

## Summary

Phase 1 of the IQGD (Iterative Q-Guided Diffusion) project has been successfully completed. The project infrastructure is set up, MGDM dependencies are integrated, and all core modules have been implemented.

---

## Completed Tasks

### 1. ✓ Environment Setup
- **Server**: Great Lakes HPC (University of Michigan)
- **Partition**: `gpu` (with GPU support)
- **Account**: `alkontar0`
- **Environment**: Python 3.11 + PyTorch 2.1.0 + CUDA 11.8
- **Modules loaded via**: `/home/rongbo/env_rl.sh`

### 2. ✓ MGDM Dependencies Copied
Copied from `~/RLresearch/MGDM/`:
- `architectures.py` - U-Net and VAE architectures
- `attend.py` - Attention mechanisms
- `utils.py` - Utility functions
- `evaluations.py` - Evaluation metrics (MSE, PSNR, SSIM)
- `ditmodels.py` - DiT model architectures

### 3. ✓ Dataset Linked
- **Symbolic link**: `data/` → `~/RLresearch/MGDM/fluid_data_gen/`
- **Dataset format**: Fluid dynamics simulations
- **Components**:
  - `cs1_list`: Coarse physics context (2 channels, 64×64)
  - `cs2_list`: Fine physics context (1 channel, 32×32)
  - `history_list`: Initial conditions (4 channels, 256×256)
  - `data_list`: Target data (1 channel, 256×256)
  - `temps_list`: Physical timestamps

### 4. ✓ Pretrained Model Copied
- **Source**: `~/RLresearch/MGDM/model_fluid_smoke__unet_200.pth`
- **Destination**: `models/pretrained/model_fluid_smoke__unet_200.pth`
- **Architecture**: U-Net with 7 input channels, 32 base dimensions, 1 output channel
- **Training**: 200 epochs on MGDM fluid dataset

### 5. ✓ IQGD Core Modules Implemented

#### `iqgd/data_loader.py`
- Wraps MGDM's fluid dataset for RL training
- Handles data loading, normalization
- Creates train/test dataloaders
- **Key functions**: `read_raw_data()`, `normalize_data()`, `create_dataloaders()`

#### `iqgd/diffusion_env.py`
- RL environment wrapping diffusion sampling process
- **State**: (x_t, x_0_pred, timestep, physics contexts)
- **Action**: Guidance strength [0, 1]
- **Reward**: PSNR + physics consistency - step penalty
- **Methods**: `reset()`, `step()`, `get_state()`

#### `iqgd/q_network.py`
- Q-network architecture for predicting guidance actions
- **Input**: 9-channel state (x_t + x_0_pred + contexts)
- **Output**: Q-values for discrete actions
- **Architecture**: CNN encoder + FC layers
- Supports both discrete and continuous action spaces

#### `iqgd/replay_buffer.py`
- Experience replay buffer for storing transitions
- Supports both standard and prioritized replay
- Efficient batching and sampling
- **Capacity**: 10,000 transitions (configurable)

#### `iqgd/iqgd_agent.py`
- Main IQGD agent implementing Q-learning
- **Algorithm**: Deep Q-Network (DQN) with target network
- **Features**:
  - Epsilon-greedy exploration
  - Target network updates
  - Experience replay
  - Checkpointing

---

## File Structure

```
IQGD/
├── iqgd/                           # Core IQGD modules
│   ├── __init__.py
│   ├── data_loader.py              # Dataset wrapper
│   ├── diffusion_env.py            # RL environment
│   ├── q_network.py                # Q-function
│   ├── iqgd_agent.py               # Main agent
│   └── replay_buffer.py            # Experience replay
│
├── models/
│   ├── pretrained/                 # Pretrained diffusion models
│   │   └── model_fluid_smoke__unet_200.pth
│   └── checkpoints/                # IQGD checkpoints (empty)
│
├── data/                           # → symlink to MGDM dataset
│
├── outputs/                        # Generated outputs
│   ├── baseline_test/             # Baseline test results
│   └── training/                   # Training outputs
│
├── logs/                           # Training and job logs
│
├── configs/                        # Configuration files (empty)
├── experiments/                    # Experiment notebooks (empty)
├── utils/                          # Utility scripts (empty)
│
├── architectures.py                # From MGDM
├── attend.py                       # From MGDM
├── ditmodels.py                    # From MGDM
├── evaluations.py                  # From MGDM
├── utils.py                        # From MGDM
│
├── test_baseline.py                # Test MGDM baseline
├── train_iqgd.py                   # IQGD training script
│
├── submit_test_baseline.slurm      # SLURM: Test baseline
├── submit_train_iqgd.slurm         # SLURM: Train IQGD
│
├── README.md                       # Project README
├── requirements.txt                # Python dependencies
├── SETUP_INFO.md                   # Setup guide
└── PHASE1_COMPLETE.md              # This file
```

---

## Scripts Created

### Testing Scripts

#### 1. `test_baseline.py`
Tests the pretrained MGDM diffusion model on fluid dataset
```bash
python test_baseline.py
```
**Outputs**:
- Baseline performance metrics (MSE, PSNR, SSIM)
- Visualizations in `outputs/baseline_test/`

#### 2. `submit_test_baseline.slurm`
SLURM job for testing baseline
```bash
sbatch submit_test_baseline.slurm
```

### Training Scripts

#### 3. `train_iqgd.py`
Main training script for IQGD agent
```bash
python train_iqgd.py
```
**Features**:
- Trains Q-network for diffusion guidance
- Periodic evaluation and checkpointing
- TensorBoard logging
- Saves models to `models/checkpoints/`

#### 4. `submit_train_iqgd.slurm`
SLURM job for IQGD training
```bash
sbatch submit_train_iqgd.slurm
```

---

## Next Steps (Phase 2)

### Immediate Actions
1. **Test baseline model**:
   ```bash
   sbatch submit_test_baseline.slurm
   ```
   - Verify pretrained model works
   - Establish baseline metrics
   - Check data loading pipeline

2. **Debug and refine**:
   - Fix any import errors
   - Validate diffusion environment
   - Test Q-network forward pass

3. **Small-scale training**:
   - Run training for 100 episodes locally
   - Verify Q-learning loop works
   - Check reward convergence

### Phase 2 Goals (Week 2)
- [ ] Complete baseline testing and validation
- [ ] Debug IQGD pipeline end-to-end
- [ ] Run initial training experiments
- [ ] Tune hyperparameters (learning rate, epsilon decay, reward weights)
- [ ] Implement evaluation metrics tracking
- [ ] Add visualization tools for Q-values and guidance

### Phase 3 Goals (Week 3-4)
- [ ] Compare IQGD vs. baseline vs. physics-guided
- [ ] Ablation studies (Q-network architecture, action space, reward design)
- [ ] Optimize for computational efficiency
- [ ] Write experiment report

---

## Key Design Decisions

### 1. Action Space
**Chosen**: Discrete actions (10 guidance strengths from 0 to 1)
- Simpler to implement and debug
- Works well with DQN
- Can upgrade to continuous (DDPG/SAC) later

### 2. Reward Function
```python
reward = w_psnr * PSNR(x_final, target)
         - w_physics * physics_loss(x_final, cs2)
         - w_step * num_steps
```
- Terminal reward for final sample quality
- Physics consistency term
- Small step penalty to encourage efficiency

### 3. Q-Network Architecture
- CNN-based encoder for spatial features
- Timestep embedding for temporal conditioning
- ~2M parameters (lightweight)
- Separate processing for different context scales

### 4. Training Strategy
**Offline-to-Online**:
1. Start with experience from random policy
2. Train Q-network on collected data
3. Gradually increase exploitation (decrease ε)
4. Fine-tune with online experience

---

## Technical Notes

### Server Configuration
- **GPU**: 1x NVIDIA GPU (varies by node)
- **Memory**: 32-64 GB RAM
- **CPUs**: 4-8 cores
- **Time limit**: 12 hours (can request up to 14 days)

### Environment Variables
```bash
SLURM_JOB_ID        # Auto-set by SLURM
CUDA_VISIBLE_DEVICES=0
```

### Data Paths
- **Home**: `/home/rongbo/RLresearch/IQGD/`
- **Scratch**: `/scratch/alkontar_root/alkontar0/rongbo/${SLURM_JOB_ID}/`
- **Dataset**: `~/RLresearch/MGDM/fluid_data_gen/new_dataset/`

---

## Git Status

All Phase 1 files are ready to be committed:
```bash
cd ~/RLresearch/IQGD
git add .
git commit -m "Phase 1 complete: Project setup and core modules"
git push
```

---

## Contact

- **Project**: IQGD (Iterative Q-Guided Diffusion)
- **Author**: Bradley-Zhu (rongbo@umich.edu)
- **Repository**: https://github.com/Bradley-Zhu/IQGD
- **Date**: January 11, 2026

---

## Acknowledgments

- Based on MGDM (Multi-Fidelity Guided Diffusion Models)
- Fluid dataset from MGDM project
- Great Lakes HPC resources from University of Michigan
