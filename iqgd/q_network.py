"""
Q-Network Architecture for IQGD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class QNetwork(nn.Module):
    """
    Q-Network that predicts Q-values or guidance actions
    Takes state (x_t, x_0_pred, context) and outputs Q-values for actions
    """

    def __init__(
        self,
        state_channels: int = 9,  # x_t(1) + x_0_pred(1) + cs1(2) + cs2(1) + init(4)
        hidden_dim: int = 64,
        num_actions: int = 10,  # Discrete action space for guidance strength
        image_size: int = 128,
        use_timestep_embedding: bool = True
    ):
        """
        Args:
            state_channels: Total channels in state representation
            hidden_dim: Hidden dimension for feature extraction
            num_actions: Number of discrete actions (guidance strengths)
            image_size: Input image size
            use_timestep_embedding: Whether to use timestep embedding
        """
        super().__init__()

        self.state_channels = state_channels
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.use_timestep_embedding = use_timestep_embedding

        # Timestep embedding (if used)
        if use_timestep_embedding:
            self.time_embed = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Encoder: Extract features from state
        self.encoder = nn.Sequential(
            # Input: state_channels x 128 x 128
            nn.Conv2d(state_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),

            # 64 x 128 x 128
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),

            # 128 x 64 x 64
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.SiLU(),

            # 256 x 32 x 32
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 8),
            nn.SiLU(),

            # 512 x 16 x 16
        )

        # Global pooling and FC layers
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        fc_input_dim = hidden_dim * 8 * 4 * 4
        if use_timestep_embedding:
            fc_input_dim += hidden_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_actions)
        )

    def forward(
        self,
        state_dict: dict,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            state_dict: Dictionary containing state components
                - x_t: Current noisy sample (B, 1, H, W)
                - x_0_pred: Predicted clean sample (B, 1, H, W)
                - physics_context_cs1: (B, 2, 64, 64)
                - physics_context_cs2: (B, 1, 32, 32)
                - init_context: (B, 4, 256, 256)
                - timestep: Current timestep (scalar or B,)
            return_features: Whether to return intermediate features

        Returns:
            Q-values for each action (B, num_actions)
        """
        # Extract components
        x_t = state_dict['x_t']  # (B, 1, 128, 128)
        x_0_pred = state_dict['x_0_pred']  # (B, 1, 128, 128)
        cs1 = state_dict['physics_context_cs1']  # (B, 2, 64, 64)
        cs2 = state_dict['physics_context_cs2']  # (B, 1, 32, 32)
        init_ctx = state_dict['init_context']  # (B, 4, 256, 256)
        timestep = state_dict['timestep']

        batch_size = x_t.shape[0]

        # Resize contexts to match 128x128
        cs1_resized = F.interpolate(cs1, size=(128, 128), mode='bilinear', align_corners=False)
        cs2_resized = F.interpolate(cs2, size=(128, 128), mode='bilinear', align_corners=False)
        init_ctx_resized = F.interpolate(init_ctx, size=(128, 128), mode='bilinear', align_corners=False)

        # Concatenate all state components
        state_concat = torch.cat([
            x_t,
            x_0_pred,
            cs1_resized,
            cs2_resized,
            init_ctx_resized
        ], dim=1)  # (B, 9, 128, 128)

        # Encode state
        features = self.encoder(state_concat)  # (B, hidden*8, 16, 16)
        features_pooled = self.pool(features)  # (B, hidden*8, 4, 4)
        features_flat = features_pooled.view(batch_size, -1)  # (B, hidden*8*4*4)

        # Add timestep embedding if used
        if self.use_timestep_embedding:
            if isinstance(timestep, int):
                timestep = torch.tensor([timestep], dtype=torch.float32, device=x_t.device)
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)
            timestep = timestep.view(-1, 1).float()
            time_emb = self.time_embed(timestep / 1000.0)  # Normalize to [0, 1]
            features_flat = torch.cat([features_flat, time_emb], dim=-1)

        # Predict Q-values
        q_values = self.fc_layers(features_flat)  # (B, num_actions)

        if return_features:
            return q_values, features_flat

        return q_values

    def select_action(
        self,
        state_dict: dict,
        epsilon: float = 0.0,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select action using epsilon-greedy policy

        Args:
            state_dict: State dictionary
            epsilon: Exploration probability
            deterministic: If True, always select greedy action

        Returns:
            Selected action index (B,)
        """
        batch_size = state_dict['x_t'].shape[0]

        if deterministic or torch.rand(1).item() > epsilon:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state_dict)
                action = q_values.argmax(dim=-1)
        else:
            # Random action
            action = torch.randint(0, self.num_actions, (batch_size,), device=state_dict['x_t'].device)

        return action


class ContinuousQNetwork(nn.Module):
    """
    Q-Network for continuous action space (similar to DDPG critic)
    Q(s, a) → scalar value
    """

    def __init__(
        self,
        state_channels: int = 9,
        action_dim: int = 1,
        hidden_dim: int = 64,
        image_size: int = 128
    ):
        super().__init__()

        # State encoder (same as discrete version)
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )

        # Combined Q-value prediction
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 4 * 4 + hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, state_dict: dict, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a)

        Args:
            state_dict: State dictionary
            action: Action tensor (B, action_dim)

        Returns:
            Q-value (B, 1)
        """
        # Prepare state (same as discrete version)
        x_t = state_dict['x_t']
        x_0_pred = state_dict['x_0_pred']
        cs1 = F.interpolate(state_dict['physics_context_cs1'], size=(128, 128), mode='bilinear', align_corners=False)
        cs2 = F.interpolate(state_dict['physics_context_cs2'], size=(128, 128), mode='bilinear', align_corners=False)
        init_ctx = F.interpolate(state_dict['init_context'], size=(128, 128), mode='bilinear', align_corners=False)

        state_concat = torch.cat([x_t, x_0_pred, cs1, cs2, init_ctx], dim=1)

        # Encode state
        state_features = self.state_encoder(state_concat)
        state_features = self.pool(state_features).view(x_t.shape[0], -1)

        # Encode action
        action_features = self.action_encoder(action)

        # Combine and predict Q-value
        combined = torch.cat([state_features, action_features], dim=-1)
        q_value = self.q_head(combined)

        return q_value


if __name__ == "__main__":
    print("Q-Network module - testing...")

    # Test discrete Q-network
    q_net = QNetwork(num_actions=10, hidden_dim=32)
    print(f"Discrete Q-Network parameters: {sum(p.numel() for p in q_net.parameters()):,}")

    # Create dummy state
    batch_size = 4
    dummy_state = {
        'x_t': torch.randn(batch_size, 1, 128, 128),
        'x_0_pred': torch.randn(batch_size, 1, 128, 128),
        'physics_context_cs1': torch.randn(batch_size, 2, 64, 64),
        'physics_context_cs2': torch.randn(batch_size, 1, 32, 32),
        'init_context': torch.randn(batch_size, 4, 256, 256),
        'timestep': 500
    }

    q_values = q_net(dummy_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0]}")

    action = q_net.select_action(dummy_state, epsilon=0.1)
    print(f"Selected actions: {action}")

    print("\nQ-Network test successful!")
