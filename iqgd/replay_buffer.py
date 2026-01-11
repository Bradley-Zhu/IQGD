"""
Replay Buffer for IQGD
Stores transitions (state, action, reward, next_state, done) for experience replay
"""

import torch
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """

    def __init__(self, capacity: int = 10000, device: str = 'cuda'):
        """
        Args:
            capacity: Maximum number of transitions to store
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: dict,
        action: torch.Tensor,
        reward: float,
        next_state: dict,
        done: bool
    ):
        """
        Add a transition to the buffer

        Args:
            state: Current state dictionary
            action: Action taken
            reward: Reward received
            next_state: Next state dictionary (None if done)
            done: Whether episode terminated
        """
        # Move tensors to CPU for storage efficiency
        state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else action

        if next_state is not None:
            next_state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in next_state.items()}
        else:
            next_state_cpu = None

        transition = (state_cpu, action_cpu, reward, next_state_cpu, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)

        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Batch states
        batch_states = self._batch_states(states)

        # Batch actions
        batch_actions = torch.stack([a for a in actions]).to(self.device)

        # Batch rewards
        batch_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Batch next_states (handle None for terminal states)
        batch_next_states = self._batch_states([s for s in next_states if s is not None])

        # Batch dones
        batch_dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def _batch_states(self, states):
        """
        Convert list of state dicts to batched state dict

        Args:
            states: List of state dictionaries

        Returns:
            Batched state dictionary
        """
        if not states:
            return None

        batched = {}
        for key in states[0].keys():
            if isinstance(states[0][key], torch.Tensor):
                batched[key] = torch.stack([s[key] for s in states]).to(self.device)
            else:
                # For scalar values like timestep
                batched[key] = torch.tensor([s[key] for s in states], device=self.device)

        return batched

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from buffer"""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (optional advanced feature)
    Samples transitions based on their TD error
    """

    def __init__(self, capacity: int = 10000, device: str = 'cuda', alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Maximum buffer size
            device: Device for tensors
            alpha: Prioritization exponent
            beta: Importance sampling exponent
        """
        super().__init__(capacity, device)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add transition with maximum priority"""
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int):
        """Sample batch based on priorities"""
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Batch states
        batch_states = self._batch_states(states)
        batch_actions = torch.stack([a for a in actions]).to(self.device)
        batch_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        batch_next_states = self._batch_states([s for s in next_states if s is not None])
        batch_dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    print("ReplayBuffer module - testing...")

    buffer = ReplayBuffer(capacity=100)

    # Add dummy transitions
    for i in range(10):
        state = {
            'x_t': torch.randn(1, 1, 128, 128),
            'x_0_pred': torch.randn(1, 1, 128, 128),
            'physics_context_cs1': torch.randn(1, 2, 64, 64),
            'physics_context_cs2': torch.randn(1, 1, 32, 32),
            'init_context': torch.randn(1, 4, 256, 256),
            'timestep': i
        }
        action = torch.tensor([0])
        reward = float(i)
        next_state = state.copy() if i < 9 else None
        done = (i == 9)

        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    # Sample batch
    if len(buffer) >= 4:
        batch = buffer.sample(4)
        states, actions, rewards, next_states, dones = batch
        print(f"Sampled batch - states x_t shape: {states['x_t'].shape}")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")

    print("\nReplayBuffer test successful!")
