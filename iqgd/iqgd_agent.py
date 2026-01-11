"""
IQGD Agent: Main agent implementing Q-learning for diffusion guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import copy
from typing import Dict, Optional, Tuple
import numpy as np

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer
from .diffusion_env import DiffusionEnv


class IQGDAgent:
    """
    Iterative Q-Guided Diffusion Agent

    Learns to guide diffusion sampling using Q-learning
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        beta_schedule: torch.Tensor,
        device: str = 'cuda',
        q_network_params: Optional[Dict] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 10000,
        target_update_freq: int = 100,
        buffer_capacity: int = 10000,
        batch_size: int = 32
    ):
        """
        Args:
            diffusion_model: Pretrained diffusion model (frozen)
            beta_schedule: Diffusion beta schedule
            device: Device to run on
            q_network_params: Parameters for Q-network initialization
            learning_rate: Learning rate for Q-network
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps for epsilon decay
            target_update_freq: Frequency of target network updates
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
        """
        self.device = device

        # Environment
        self.env = DiffusionEnv(
            diffusion_model=diffusion_model,
            beta_schedule=beta_schedule,
            device=device,
            action_type='strength'
        )

        # Q-networks
        q_network_params = q_network_params or {}
        self.q_network = QNetwork(**q_network_params).to(device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.eval()

        # Optimizer
        self.optimizer = AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, device=device)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        # Training state
        self.steps = 0
        self.episodes = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon for epsilon-greedy exploration"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps / self.epsilon_decay)

    def sample_episode(
        self,
        target: torch.Tensor,
        init_context: torch.Tensor,
        physics_context_cs1: torch.Tensor,
        physics_context_cs2: torch.Tensor,
        phs_time: torch.Tensor,
        train: bool = True
    ) -> Tuple[torch.Tensor, float, Dict]:
        """
        Run one episode of Q-guided diffusion sampling

        Args:
            target: Ground truth target
            init_context: Initial conditions
            physics_context_cs1: Coarse physics context
            physics_context_cs2: Fine physics context
            phs_time: Physical timestamps
            train: Whether to train (add to replay buffer)

        Returns:
            final_sample: Generated sample
            total_reward: Cumulative reward
            info: Additional information
        """
        # Reset environment
        state = self.env.reset(target, init_context, physics_context_cs1, physics_context_cs2, phs_time)

        total_reward = 0
        trajectory = []
        done = False

        while not done:
            # Select action
            action_idx = self.q_network.select_action(
                state,
                epsilon=self.epsilon if train else 0.0,
                deterministic=not train
            )

            # Convert discrete action to continuous guidance strength
            # Map action_idx [0, num_actions-1] to strength [0, 1]
            num_actions = self.q_network.num_actions
            strength = action_idx.float() / (num_actions - 1)
            action = strength.unsqueeze(-1)  # (B, 1)

            # Take step in environment
            next_state, reward, done, step_info = self.env.step(action)

            # Store transition
            if train:
                self.replay_buffer.add(state, action_idx, reward, next_state, done)

            total_reward += reward
            trajectory.append(step_info)
            state = next_state

            if not done:
                state = next_state

        final_sample = self.env.x_t

        info = {
            'trajectory': trajectory,
            'num_steps': len(trajectory)
        }

        self.episodes += 1

        return final_sample, total_reward, info

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (Q-learning update)

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q-values
        q_values = self.q_network(states)  # (B, num_actions)
        q_values = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)  # (B,)

        # Compute target Q-values
        with torch.no_grad():
            if next_states is not None:
                next_q_values = self.target_q_network(next_states)  # (B, num_actions)
                next_q_values = next_q_values.max(dim=1)[0]  # (B,)
            else:
                next_q_values = torch.zeros_like(rewards)

            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"Agent loaded from {path}")


if __name__ == "__main__":
    print("IQGDAgent module - skeleton implementation complete")
    print("To test, initialize with a pretrained diffusion model")
