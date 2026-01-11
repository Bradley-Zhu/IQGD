"""
Test MGDM Baseline Diffusion Model
Load pretrained model and test on fluid dataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from architectures import Unet
from iqgd.data_loader import create_dataloaders
from evaluations import mse_psnr_ssim
import matplotlib.pyplot as plt
import numpy as np


def load_pretrained_diffusion_model(model_path, device='cuda'):
    """
    Load pretrained MGDM diffusion model

    Args:
        model_path: Path to model checkpoint
        device: Device to load on

    Returns:
        Loaded model
    """
    # Create model architecture (same as MGDM)
    in_channels = 7  # x_t(1) + physics_context(2) + init_context(4)
    out_channels = 1
    model = Unet(channels=in_channels, dim=32, out_dim=out_channels).to(device)

    # Load weights
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully!")
    return model


def sample_diffusion(
    model,
    initial_context,
    physics_context,
    target_data,
    phs_time,
    T=1000,
    beta_start=1e-4,
    beta_end=0.02,
    device='cuda'
):
    """
    Sample from diffusion model (DDPM sampling)

    Args:
        model: Diffusion model
        initial_context: Initial conditions (B, 4, 256, 256)
        physics_context: Physics context (B, 2, 64, 64)
        target_data: Target data for evaluation (B, 1, 256, 256)
        phs_time: Physical timestamps
        T: Number of diffusion steps
        beta_start: Start of beta schedule
        beta_end: End of beta schedule
        device: Device

    Returns:
        Generated sample, evaluation metrics
    """
    batch_size = initial_context.shape[0]

    # Diffusion schedule
    beta_list = torch.linspace(beta_start, beta_end, steps=T, device=device)
    alpha_list = 1 - beta_list
    alpha_bar_list = torch.cumprod(alpha_list, dim=0)

    # Preprocessing (same as MGDM)
    with torch.no_grad():
        # Encode contexts
        initial_context_encoded = F.avg_pool2d(initial_context, kernel_size=2, stride=2)
        physics_context_encoded = F.interpolate(
            physics_context,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )
        target_encoded = F.avg_pool2d(target_data, kernel_size=2, stride=2)

    # Start from pure noise
    x_t = torch.randn_like(target_encoded)

    # Reverse diffusion process
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            t_tensor = torch.tensor([t], device=device)

            # Prepare input
            model_input = torch.cat([
                x_t,
                physics_context_encoded,
                initial_context_encoded
            ], dim=1)

            # Predict noise
            epsilon = model(
                model_input,
                t_tensor.repeat(batch_size),
                phs_time[:, 0]
            )

            # DDPM sampling step
            alpha_bar_t = alpha_bar_list[t]
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon) / torch.sqrt(alpha_bar_t)

            if t > 0:
                alpha_bar_t_minus_1 = alpha_bar_list[t - 1]
                x_t = torch.sqrt(alpha_bar_t_minus_1) * x_0_pred + \
                      torch.sqrt(1 - alpha_bar_t_minus_1) * epsilon
            else:
                x_t = x_0_pred

    # Evaluate
    result_dict, result_dict_std = mse_psnr_ssim(target_encoded, x_t)

    return x_t, result_dict


def visualize_samples(samples, targets, save_path):
    """
    Visualize generated samples vs targets

    Args:
        samples: Generated samples (B, 1, H, W)
        targets: Target data (B, 1, H, W)
        save_path: Path to save visualization
    """
    num_samples = min(4, samples.shape[0])

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))

    for i in range(num_samples):
        # Target
        axes[0, i].imshow(targets[i, 0].cpu().numpy(), cmap='coolwarm')
        axes[0, i].set_title(f'Target {i}')
        axes[0, i].axis('off')

        # Generated
        axes[1, i].imshow(samples[i, 0].cpu().numpy(), cmap='coolwarm')
        axes[1, i].set_title(f'Generated {i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def main():
    """Main testing function"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model_path = 'models/pretrained/model_fluid_smoke__unet_200.pth'
    model = load_pretrained_diffusion_model(model_path, device)

    # Load dataset
    print("\nLoading dataset...")
    train_loader, test_loader, train_ds, test_ds = create_dataloaders(
        data_path='data/new_dataset',
        batch_size=4
    )

    # Test on a few batches
    print("\nTesting baseline diffusion sampling...")
    all_metrics = []

    os.makedirs('outputs/baseline_test', exist_ok=True)

    for i, batch in enumerate(test_loader):
        if i >= 3:  # Test on 3 batches
            break

        init_ctx, cs1, cs2, target, phs_time = batch
        init_ctx = init_ctx.to(device)
        cs1 = cs1.to(device)
        cs2 = cs2.to(device)
        target = target.to(device)
        phs_time = phs_time.to(device)

        print(f"\nBatch {i + 1}:")
        print(f"  Initial context: {init_ctx.shape}")
        print(f"  Physics context (cs1): {cs1.shape}")
        print(f"  Physics context (cs2): {cs2.shape}")
        print(f"  Target: {target.shape}")

        # Sample
        samples, metrics = sample_diffusion(
            model, init_ctx, cs1, target, phs_time, device=device
        )

        print(f"  Metrics: MSE={metrics[0]:.6f}, PSNR={metrics[1]:.2f}, SSIM={metrics[2]:.4f}")
        all_metrics.append(metrics)

        # Visualize
        target_encoded = F.avg_pool2d(target, kernel_size=2, stride=2)
        visualize_samples(
            samples,
            target_encoded,
            f'outputs/baseline_test/batch_{i+1}.png'
        )

    # Summary
    all_metrics = torch.stack(all_metrics)
    mean_metrics = all_metrics.mean(dim=0)
    std_metrics = all_metrics.std(dim=0)

    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"MSE:  {mean_metrics[0]:.6f} ± {std_metrics[0]:.6f}")
    print(f"PSNR: {mean_metrics[1]:.2f} ± {std_metrics[1]:.2f}")
    print(f"SSIM: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
