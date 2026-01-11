# IQGD: Iterative Q-Guided Diffusion

## Overview
Iterative Q-Guided Diffusion (IQGD) is a reinforcement learning research project that combines Q-learning with diffusion models for improved policy learning.

## Project Structure
```
IQGD/
├── configs/          # Configuration files for experiments
├── experiments/      # Experiment scripts and notebooks
├── logs/            # Training logs and metrics
├── models/          # Saved model checkpoints
├── outputs/         # Generated outputs and visualizations
├── utils/           # Utility functions and helpers
├── iqgd_agent.py    # Main IQGD agent implementation
├── train.py         # Training script
├── eval.py          # Evaluation script
└── requirements.txt # Project dependencies
```

## Setup
```bash
# Create and activate conda environment
conda create -n iqgd python=3.9
conda activate iqgd

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Train the IQGD agent
python train.py --config configs/default.yaml

# Evaluate trained model
python eval.py --model models/checkpoint.pth
```

## Citation
TBD

## License
TBD
