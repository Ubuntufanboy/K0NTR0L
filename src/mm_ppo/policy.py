"""
PPO Actor-Critic Network Architecture
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Policy network (Actor) with GRU backbone"""
    
    def __init__(self, 
                 input_dim: int,
                 action_dim: int,
                 text_dim: int,
                 vision_dim: int,
                 sensor_dim: int, # Kept for API compatibility, but ignored
                 hidden_dims: list = [1024], # Kept for API compatibility
                 continuous: bool = False,
                 n_heads: int = 4): # Kept for API compatibility
        super().__init__()
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        
        # GRU Backbone
        # Input is Text + Vision
        self.gru_input_dim = text_dim + vision_dim
        self.hidden_dim = 1024
        self.num_layers = 2
        
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Heads
        if continuous:
            self.mean_head = nn.Linear(self.hidden_dim, action_dim)
            self.log_std_head = nn.Linear(self.hidden_dim, action_dim)
        else:
            self.action_head = nn.Linear(self.hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"GRU Policy network initialized: "
                   f"Vision({vision_dim}) + Text({text_dim}) -> "
                   f"GRU({self.hidden_dim}, {self.num_layers} layers) -> {action_dim}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, hidden=None):
        # x shape: (Batch, Seq, Dim) or (Batch, Dim)
        
        # If input is (Batch, Dim), unsqueeze to (Batch, 1, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Split input and ignore sensor data
        # Assuming x is [Text | Vision | Sensor]
        text_emb = x[:, :, :self.text_dim]
        vision_emb = x[:, :, self.text_dim : self.text_dim + self.vision_dim]
        # sensor_emb = x[:, :, self.text_dim + self.vision_dim:] # Ignored
        
        # Concatenate Text + Vision
        gru_input = torch.cat([text_emb, vision_emb], dim=-1)
        
        # Pass through GRU
        output, new_hidden = self.gru(gru_input, hidden)
        
        # Output shape: (Batch, Seq, Hidden)
        
        if self.continuous:
            mean = self.mean_head(output)
            log_std = self.log_std_head(output)
            log_std = torch.clamp(log_std, -20, 2)
            return mean, log_std, new_hidden
        else:
            logits = self.action_head(output)
            return logits, new_hidden
    
    def get_action_and_log_prob(self, x, hidden=None, deterministic=False):
        """Sample action and return log probability"""
        
        if self.continuous:
            mean, log_std, new_hidden = self.forward(x, hidden)
            std = log_std.exp()
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            
            # Log prob
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action, log_prob, new_hidden
        else:
            logits, new_hidden = self.forward(x, hidden)
            dist = Categorical(logits=logits)
            
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            return action, log_prob, new_hidden
    
    def evaluate_actions(self, x, actions, hidden=None):
        """Evaluate log probability and entropy of actions"""
        
        if self.continuous:
            mean, log_std, _ = self.forward(x, hidden)
            std = log_std.exp()
            dist = Normal(mean, std)
            
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits, _ = self.forward(x, hidden)
            dist = Categorical(logits=logits)
            
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value network (Critic) for PPO"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [1024, 512]):
        super().__init__()
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Value network initialized: {input_dim} -> {hidden_dims} -> 1")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        features = self.backbone(x)
        value = self.value_head(features)
        return value.squeeze(-1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic for PPO"""
    
    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 text_dim: int,
                 vision_dim: int,
                 sensor_dim: int,
                 hidden_dims: list = [1024, 512],
                 continuous: bool = False):
        super().__init__()
        
        self.policy = PolicyNetwork(
            observation_dim, 
            action_dim, 
            text_dim,
            vision_dim,
            sensor_dim,
            hidden_dims, 
            continuous
        )
        self.value = ValueNetwork(observation_dim, hidden_dims)
        
        self.continuous = continuous
    
    def get_action(self, obs, hidden=None, deterministic=False):
        """Get action from observation"""
        action, log_prob, new_hidden = self.policy.get_action_and_log_prob(obs, hidden, deterministic)
        value = self.value(obs)
        return action, log_prob, value, new_hidden
    
    def evaluate(self, obs, actions, hidden=None):
        """Evaluate actions"""
        log_prob, entropy = self.policy.evaluate_actions(obs, actions, hidden)
        value = self.value(obs)
        return log_prob, entropy, value
    
    def get_value(self, obs):
        """Get value estimate"""
        return self.value(obs)


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, observation_dim: int, action_dim: int, 
                 device: str = "cuda", continuous: bool = False):
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.continuous = continuous
        
        # Allocate buffers
        self.observations = torch.zeros((buffer_size, observation_dim), device=device)
        
        if continuous:
            self.actions = torch.zeros((buffer_size, action_dim), device=device)
        else:
            self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs, action, reward, value, log_prob, done):
        """Add experience to buffer"""
        
        self.observations[self.ptr] = obs.detach()
        self.actions[self.ptr] = action.detach()
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = value.detach()
        self.log_probs[self.ptr] = log_prob.detach()
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def get(self):
        """Get all data and compute advantages"""
        
        size = self.buffer_size if self.full else self.ptr
        
        return {
            'observations': self.observations[:size],
            'actions': self.actions[:size],
            'rewards': self.rewards[:size],
            'values': self.values[:size],
            'log_probs': self.log_probs[:size],
            'dones': self.dones[:size]
        }
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.full = False
    
    def __len__(self):
        return self.buffer_size if self.full else self.ptr


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation"""
    
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns
