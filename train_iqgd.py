"""
Training Script for IQGD
Train Q-network for Q-guided diffusion
"""

import torch
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from architectures import Unet
from iqgd import IQGDAgent, create_dataloaders
from evaluations import mse_psnr_ssim


def train_iqgd(
    num_episodes=1000,
    batch_size=16,
    learning_rate=1e-4,
    device='cuda',
    save_freq=50,
    eval_freq=10,
    log_dir='logs/iqgd_training'
):
    """
    Train IQGD agent

    Args:
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
        save_freq: Frequency of saving checkpoints
        eval_freq: Frequency of evaluation
        log_dir: Directory for logs
    """
    print("=" * 60)
    print("IQGD Training")
    print("=" * 60)

    # Create directories
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('outputs/training', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Load pretrained diffusion model
    print("\nLoading pretrained diffusion model...")
    model_path = 'models/pretrained/model_fluid_smoke__unet_200.pth'
    diffusion_model = Unet(channels=7, dim=32, out_dim=1).to(device)
    diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
    diffusion_model.eval()
    print("Diffusion model loaded!")

    # Create dataset
    print("\nLoading dataset...")
    train_loader, test_loader, train_ds, test_ds = create_dataloaders(
        data_path='data/new_dataset',
        batch_size=batch_size
    )

    # Create IQGD agent
    print("\nInitializing IQGD agent...")
    beta_schedule = torch.linspace(1e-4, 0.02, steps=1000, device=device)
    agent = IQGDAgent(
        diffusion_model=diffusion_model,
        beta_schedule=beta_schedule,
        device=device,
        q_network_params={'hidden_dim': 64, 'num_actions': 10},
        learning_rate=learning_rate,
        buffer_capacity=10000,
        batch_size=32
    )
    print("Agent initialized!")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    train_data_iter = iter(train_loader)
    episode_rewards = []
    losses = []

    for episode in range(num_episodes):
        # Get batch
        try:
            batch = next(train_data_iter)
        except StopIteration:
            train_data_iter = iter(train_loader)
            batch = next(train_data_iter)

        init_ctx, cs1, cs2, target, phs_time = batch
        init_ctx = init_ctx.to(device)
        cs1 = cs1.to(device)
        cs2 = cs2.to(device)
        target = target.to(device)
        phs_time = phs_time.to(device)

        # Sample episode for each item in batch (for simplicity, use first item)
        sample_idx = 0
        _, reward, info = agent.sample_episode(
            target[sample_idx:sample_idx+1],
            init_ctx[sample_idx:sample_idx+1],
            cs1[sample_idx:sample_idx+1],
            cs2[sample_idx:sample_idx+1],
            phs_time[sample_idx:sample_idx+1],
            train=True
        )

        episode_rewards.append(reward)

        # Train Q-network
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_loss = np.mean(losses[-10:]) if losses else 0
            epsilon = agent.epsilon

            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {reward:.2f} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer)}")

            writer.add_scalar('Train/Reward', reward, episode)
            writer.add_scalar('Train/Avg_Reward', avg_reward, episode)
            writer.add_scalar('Train/Loss', avg_loss, episode)
            writer.add_scalar('Train/Epsilon', epsilon, episode)
            writer.add_scalar('Train/Buffer_Size', len(agent.replay_buffer), episode)

        # Evaluation
        if episode % eval_freq == 0 and episode > 0:
            print(f"\nEvaluating at episode {episode}...")
            eval_metrics = evaluate_iqgd(agent, test_loader, device, num_batches=3)

            print(f"Eval Metrics - PSNR: {eval_metrics['psnr']:.2f}, "
                  f"SSIM: {eval_metrics['ssim']:.4f}")

            writer.add_scalar('Eval/PSNR', eval_metrics['psnr'], episode)
            writer.add_scalar('Eval/SSIM', eval_metrics['ssim'], episode)

        # Save checkpoint
        if episode % save_freq == 0 and episode > 0:
            save_path = f'models/checkpoints/iqgd_episode_{episode}.pth'
            agent.save(save_path)
            print(f"Checkpoint saved: {save_path}")

    # Final save
    agent.save('models/checkpoints/iqgd_final.pth')
    writer.close()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Plot training curves
    plot_training_curves(episode_rewards, losses, 'outputs/training/training_curves.png')


def evaluate_iqgd(agent, test_loader, device, num_batches=5):
    """
    Evaluate IQGD agent

    Args:
        agent: IQGD agent
        test_loader: Test data loader
        device: Device
        num_batches: Number of batches to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    all_metrics = []

    for i, batch in enumerate(test_loader):
        if i >= num_batches:
            break

        init_ctx, cs1, cs2, target, phs_time = batch
        init_ctx = init_ctx.to(device)
        cs1 = cs1.to(device)
        cs2 = cs2.to(device)
        target = target.to(device)
        phs_time = phs_time.to(device)

        # Sample with agent (deterministic)
        sample_idx = 0
        final_sample, reward, info = agent.sample_episode(
            target[sample_idx:sample_idx+1],
            init_ctx[sample_idx:sample_idx+1],
            cs1[sample_idx:sample_idx+1],
            cs2[sample_idx:sample_idx+1],
            phs_time[sample_idx:sample_idx+1],
            train=False
        )

        # Compute metrics
        target_encoded = torch.nn.functional.avg_pool2d(
            target[sample_idx:sample_idx+1],
            kernel_size=2,
            stride=2
        )
        metrics, _ = mse_psnr_ssim(target_encoded, final_sample)
        all_metrics.append(metrics)

    all_metrics = torch.stack(all_metrics)
    mean_metrics = all_metrics.mean(dim=0)

    return {
        'mse': mean_metrics[0].item(),
        'psnr': mean_metrics[1].item(),
        'ssim': mean_metrics[2].item()
    }


def plot_training_curves(rewards, losses, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Rewards
    axes[0].plot(rewards)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].grid(True)

    # Losses
    axes[1].plot(losses)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Q-Network Loss')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_iqgd(
        num_episodes=1000,
        batch_size=16,
        learning_rate=1e-4,
        device=device
    )
