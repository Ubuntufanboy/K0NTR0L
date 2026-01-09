"""
Main Training Loop for Multi-Modal PPO
"""

import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from .core import (
    TaskRegistry, 
    FrameBuffer, 
    VLMRewardEvaluator,
    MetricsLogger
)
from .encoders import MultiModalEncoder
from .policy import ActorCritic, RolloutBuffer
from .trainer import MultiTaskPPOTrainer
from .env_wrapper import MultiModalEnvWrapper, HighwayEnvWrapper

logger = logging.getLogger(__name__)


class MultiModalPPOTrainingLoop:
    """
    Main training loop orchestrating all components
    """
    
    def __init__(self,
                 env_wrapper: MultiModalEnvWrapper,
                 tasks_path: str,
                 log_dir: str,
                 vllm_host: str = "localhost",
                 vllm_port: int = 8000,
                 device: str = "cuda",
                 # Training hyperparameters
                 rollout_steps: int = 2048,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 task_switch_frequency: int = 5,
                 # VLM evaluation
                 vlm_eval_frequency: int = 10,  # Evaluate every N steps
                 frame_capture_fps: int = 5,
                 vlm_reward_coef: float = 5.0,
                 # Logging
                 save_frequency: int = 100):
        
        self.env = env_wrapper
        self.device = device
        
        # Initialize components
        logger.info("Initializing training components...")
        
        # Task management
        self.task_registry = TaskRegistry(tasks_path)
        self.current_task = self.task_registry.get_current_task()
        
        # Determine kinematics dimension from environment
        kinematics_dim = np.prod(self.env.observation_space.shape)

        # Multi-modal encoder
        self.encoder = MultiModalEncoder(
            kinematics_dim=kinematics_dim,
            device=device
        )
        
        # Actor-Critic
        self.actor_critic = ActorCritic(
            observation_dim=self.encoder.total_dim,
            action_dim=self.env.action_dim,
            text_dim=self.encoder.text_dim,
            vision_dim=self.encoder.vision_dim,
            sensor_dim=self.encoder.sensor_dim,
            continuous=self.env.continuous_actions
        ).to(device)
        
        # Trainer
        self.trainer = MultiTaskPPOTrainer(
            self.actor_critic,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            n_epochs=n_epochs,
            batch_size=batch_size,
            task_switch_frequency=task_switch_frequency,
            device=device,
            additional_parameters=list(self.encoder.sensor_encoder.parameters())
        )
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=rollout_steps,
            observation_dim=self.encoder.total_dim,
            action_dim=self.env.action_dim,
            device=device,
            continuous=self.env.continuous_actions
        )
        
        # VLM evaluator
        self.vlm_evaluator = VLMRewardEvaluator(
            vllm_host=vllm_host,
            vllm_port=vllm_port
        )
        
        # Frame buffer for VLM
        self.frame_buffer = FrameBuffer(fps=frame_capture_fps)
        
        # Metrics logger
        self.metrics_logger = MetricsLogger(log_dir)
        
        # Training parameters
        self.rollout_steps = rollout_steps
        self.vlm_eval_frequency = vlm_eval_frequency
        self.vlm_reward_coef = vlm_reward_coef
        self.save_frequency = save_frequency
        
        # State
        self.global_step = 0
        self.episode_count = 0
        
        logger.info("Training loop initialized successfully")
    
    def _update_command(self):
        """Update encoder with current task command"""
        self.encoder.set_command(self.current_task.command)
        logger.info(f"Active command: {self.current_task.command}")
    
    def _should_evaluate_with_vlm(self) -> bool:
        """Check if we should run VLM evaluation"""
        return self.global_step % self.vlm_eval_frequency == 0
    
    def _run_vlm_evaluation(self) -> Optional[float]:
        """Run VLM evaluation on recent frames"""
        
        if len(self.frame_buffer.frames) < 3:
            return None
        
        # Get recent frames
        recent_frames = self.frame_buffer.get_recent_frames(n=5)
        
        # Evaluate with VLM
        try:
            evaluation = self.vlm_evaluator.evaluate(
                self.current_task,
                recent_frames
            )
            
            logger.debug(f"VLM evaluation: reward={evaluation.total_reward:.2f}, "
                        f"success={evaluation.success_score:.2f}")
            
            return evaluation.total_reward
            
        except Exception as e:
            logger.error(f"VLM evaluation failed: {e}")
            return None
    
    def run_episode(self) -> dict:
        """Run one episode"""
        
        # Reset environment
        obs_dict = self.env.reset()
        
        # Encode initial observation
        obs_encoded = self.encoder.encode_state(obs_dict)
        
        # Clear frame buffer
        self.frame_buffer.clear()
        
        episode_reward = 0.0
        episode_vlm_reward = 0.0
        episode_steps = 0
        vlm_evaluations = []
        
        done = False
        start_time = time.time()
        
        while not done and episode_steps < self.current_task.episode_length:
            
            # Capture frame if needed
            current_time = time.time()
            if self.frame_buffer.should_capture(current_time):
                self.frame_buffer.add_frame(obs_dict['frame'], current_time)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action(
                    obs_encoded.unsqueeze(0)
                )
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)
            
            # Execute action
            next_obs_dict, env_reward, done, info = self.env.step(action)
            
            # VLM evaluation
            vlm_reward = 0.0
            if self._should_evaluate_with_vlm():
                vlm_eval = self._run_vlm_evaluation()
                if vlm_eval is not None:
                    vlm_reward = vlm_eval
                    episode_vlm_reward += vlm_reward
                    
                    # Store evaluation
                    vlm_evaluations.append(self.vlm_evaluator.evaluate(
                        self.current_task,
                        self.frame_buffer.get_recent_frames(n=5)
                    ))
            
            # Combined reward (environment + VLM)
            combined_reward = env_reward + (vlm_reward * self.vlm_reward_coef)
            
            # Store in buffer
            self.rollout_buffer.add(
                obs_encoded,
                action,
                combined_reward,
                value,
                log_prob,
                float(done)
            )
            
            # Update state
            obs_dict = next_obs_dict
            obs_encoded = self.encoder.encode_state(obs_dict)
            
            episode_reward += combined_reward
            episode_steps += 1
            self.global_step += 1
            
            # Train if buffer is full
            if len(self.rollout_buffer) >= self.rollout_steps:
                train_stats = self.trainer.train_step(self.rollout_buffer)
                
                # Log training metrics
                self.metrics_logger.log_training_metrics(
                    self.global_step,
                    train_stats['policy_loss'],
                    train_stats['value_loss'],
                    train_stats['entropy']
                )
                
                # Clear buffer
                self.rollout_buffer.clear()
        
        episode_duration = time.time() - start_time
        
        # Log episode
        self.metrics_logger.log_episode(
            self.episode_count,
            self.current_task.task_id,
            episode_reward,
            episode_steps,
            vlm_evaluations
        )
        
        # Update task statistics
        self.trainer.update_task_stats(
            self.current_task.task_id,
            {
                'episode_reward': episode_reward,
                'success': vlm_evaluations[-1].success_score if vlm_evaluations else 0.0
            }
        )
        
        logger.info(f"Episode {self.episode_count} completed: "
                   f"reward={episode_reward:.2f} (env+vlm), "
                   f"steps={episode_steps}, "
                   f"duration={episode_duration:.2f}s")
        
        return {
            'episode_reward': episode_reward,
            'episode_vlm_reward': episode_vlm_reward,
            'episode_steps': episode_steps,
            'episode_duration': episode_duration,
            'vlm_evaluations': len(vlm_evaluations)
        }
    
    def train(self, n_episodes: int):
        """Main training loop"""
        
        logger.info(f"Starting training for {n_episodes} episodes")
        logger.info(f"Environment: {self.env.env_name}")
        logger.info(f"Tasks: {len(self.task_registry.tasks)}")
        
        # Initial command
        self._update_command()
        
        try:
            for episode in range(n_episodes):
                self.episode_count = episode
                
                # Check if we should switch tasks
                if self.trainer.should_switch_task():
                    self.current_task = self.task_registry.switch_task(strategy='sequential')
                    self._update_command()
                
                # Run episode
                episode_stats = self.run_episode()
                
                # Save checkpoint
                if (episode + 1) % self.save_frequency == 0:
                    checkpoint_path = Path(self.metrics_logger.log_dir) / f"checkpoint_ep{episode+1}.pt"
                    self.trainer.save_checkpoint(str(checkpoint_path))
                    
                    # Also save metrics
                    self.metrics_logger.save_checkpoint(
                        episode,
                        self.actor_critic.state_dict()
                    )
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            logger.info("Cleaning up...")
            self.metrics_logger.close()
            self.env.close()
            logger.info("Training completed")
    
    def evaluate(self, n_episodes: int = 10, render: bool = True):
        """Evaluate trained policy"""
        
        logger.info(f"Evaluating policy for {n_episodes} episodes")
        
        self.actor_critic.eval()
        
        for episode in range(n_episodes):
            obs_dict = self.env.reset()
            obs_encoded = self.encoder.encode_state(obs_dict)
            
            episode_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < self.current_task.episode_length:
                with torch.no_grad():
                    action, _, _ = self.actor_critic.get_action(
                        obs_encoded.unsqueeze(0),
                        deterministic=True
                    )
                
                next_obs_dict, reward, done, _ = self.env.step(action.squeeze(0))
                
                obs_encoded = self.encoder.encode_state(next_obs_dict)
                episode_reward += reward
                steps += 1
                
                if render:
                    self.env.env.render()
            
            logger.info(f"Eval episode {episode + 1}: reward={episode_reward:.2f}, steps={steps}")
        
        self.actor_critic.train()
