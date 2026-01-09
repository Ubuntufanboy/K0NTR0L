"""
PPO Training Algorithm Implementation
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from .policy import ActorCritic, RolloutBuffer, compute_gae

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Proximal Policy Optimization trainer"""
    
    def __init__(self,
                 actor_critic: ActorCritic,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 device: str = "cuda",
                 additional_parameters: list = None):
        
        self.actor_critic = actor_critic
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Optimizer
        params = list(actor_critic.parameters())
        if additional_parameters:
            params.extend(additional_parameters)
            
        self.optimizer = optim.Adam(
            params,
            lr=learning_rate,
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.95
        )
        
        # Training stats
        self.training_step = 0
        
        logger.info(f"PPO Trainer initialized with lr={learning_rate}, "
                   f"clip_eps={clip_epsilon}, gamma={gamma}")
    
    def train_step(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform one training step"""
        
        # Get rollout data
        data = rollout_buffer.get()
        
        observations = data['observations']
        actions = data['actions']
        old_log_probs = data['log_probs']
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']
        
        # Compute advantages and returns
        advantages, returns = compute_gae(
            rewards, values, dones,
            self.gamma, self.gae_lambda
        )
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        
        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            
            # Generate random indices for mini-batches
            indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values = self.actor_critic.evaluate(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = values
                value_loss = 0.5 * (batch_returns - value_pred_clipped).pow(2).mean()
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                # Clip fraction (measure of constraint violation)
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    clip_fractions.append(clip_fraction.item())
        
        self.training_step += 1
        self.scheduler.step()
        
        # Return training statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return stats
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'training_step': self.training_step
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_step = checkpoint['training_step']
        logger.info(f"Checkpoint loaded from {path}")


class MultiTaskPPOTrainer(PPOTrainer):
    """Extended PPO trainer with multi-task support"""
    
    def __init__(self, *args, task_switch_frequency: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.task_switch_frequency = task_switch_frequency
        self.episodes_since_switch = 0
        
        # Per-task statistics
        self.task_stats = {}
        
        logger.info(f"Multi-task PPO trainer initialized. "
                   f"Task switch frequency: {task_switch_frequency} episodes")
    
    def should_switch_task(self) -> bool:
        """Determine if we should switch tasks"""
        self.episodes_since_switch += 1
        
        if self.episodes_since_switch >= self.task_switch_frequency:
            self.episodes_since_switch = 0
            return True
        return False
    
    def update_task_stats(self, task_id: str, stats: Dict[str, Any]):
        """Update statistics for a specific task"""
        if task_id not in self.task_stats:
            self.task_stats[task_id] = {
                'episodes': 0,
                'total_reward': 0.0,
                'avg_reward': 0.0,
                'success_rate': 0.0
            }
        
        task_stat = self.task_stats[task_id]
        task_stat['episodes'] += 1
        task_stat['total_reward'] += stats.get('episode_reward', 0.0)
        task_stat['avg_reward'] = task_stat['total_reward'] / task_stat['episodes']
        
        if 'success' in stats:
            old_success = task_stat['success_rate'] * (task_stat['episodes'] - 1)
            task_stat['success_rate'] = (old_success + stats['success']) / task_stat['episodes']
    
    def get_task_performance(self, task_id: str) -> Dict[str, float]:
        """Get performance statistics for a task"""
        return self.task_stats.get(task_id, {
            'episodes': 0,
            'avg_reward': 0.0,
            'success_rate': 0.0
        })
