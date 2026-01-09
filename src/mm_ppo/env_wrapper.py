"""
Environment Wrapper for Multi-Modal PPO Training
Provides standardized interface for any Gym/Gymnasium environment
"""

import logging
import time
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional

# Gymnasium/Gym Compatibility
try:
    import gymnasium as gym
    USING_GYMNASIUM = True
except ImportError:
    import gym
    USING_GYMNASIUM = False

logger = logging.getLogger(__name__)
logger.info(f"Using {'Gymnasium' if USING_GYMNASIUM else 'Gym'}")


def setup_headless_display():
    """Setup virtual display for headless environments"""
    try:
        # Check if we're in a headless environment
        if 'DISPLAY' not in os.environ:
            logger.info("No display detected, setting up virtual display...")
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(1400, 900))
            display.start()
            logger.info("Virtual display started")
            return display
    except ImportError:
        logger.warning("pyvirtualdisplay not installed. Install with: pip install pyvirtualdisplay")
        logger.warning("Also install xvfb: sudo apt-get install xvfb")
    except Exception as e:
        logger.warning(f"Could not setup virtual display: {e}")
    return None


class MultiModalEnvWrapper:
    """
    Wrapper for Gym environments to work with multi-modal PPO
    Handles frame extraction, action mapping, and reward tracking
    """
    
    def __init__(self, 
                 env_name: str,
                 env_config: Optional[Dict[str, Any]] = None,
                 render_mode: Optional[str] = None,
                 headless: bool = None):
        
        self.env_name = env_name
        self.render_mode = render_mode
        
        # Auto-detect headless environment
        if headless is None:
            headless = 'DISPLAY' not in os.environ
        
        self.headless = headless
        self.virtual_display = None
        
        # Setup virtual display if needed
        if self.headless and render_mode == 'rgb_array':
            self.virtual_display = setup_headless_display()
        
        # Create environment
        logger.info(f"Creating environment: {env_name}")
        
        try:
            if env_config:
                self.env = gym.make(env_name, **env_config)
            else:
                self.env = gym.make(env_name)
        except TypeError as e:
            # If config fails, try without it
            logger.warning(f"Failed to create env with config: {e}")
            logger.info("Retrying without config...")
            self.env = gym.make(env_name)
        
        # Set render mode if specified
        if render_mode:
            try:
                # Gymnasium usually sets this in make(), but legacy gym might need this
                if hasattr(self.env, 'render_mode'):
                    self.env.render_mode = render_mode
            except Exception as e:
                logger.warning(f"Could not set render_mode to {render_mode}: {e}")
        
        # Get action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Determine if actions are continuous or discrete
        self.continuous_actions = isinstance(
            self.action_space, 
            gym.spaces.Box
        )
        
        # Get dimensions
        if self.continuous_actions:
            self.action_dim = self.action_space.shape[0]
            self.action_low = self.action_space.low
            self.action_high = self.action_space.high
        else:
            self.action_dim = self.action_space.n
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        
        logger.info(f"Environment initialized: {env_name}")
        logger.info(f"  Action space: {self.action_space} ({'continuous' if self.continuous_actions else 'discrete'})")
        logger.info(f"  Action dim: {self.action_dim}")
        logger.info(f"  Observation space: {self.observation_space}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation dict"""
        result = self.env.reset()
        
        # Gymnasium returns (obs, info), old gym returns just obs
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        
        # Get visual frame
        frame = self._get_frame()
        
        return {'frame': frame, 'obs': obs}
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute action and return next state dict"""
        
        # Convert action to environment format
        if self.continuous_actions:
            # Ensure action is numpy array
            if isinstance(action, (int, float)):
                action = np.array([action])
            elif hasattr(action, 'cpu'):  # PyTorch tensor
                action = action.cpu().numpy()
            
            # Clip to action bounds
            action = np.clip(action, self.action_low, self.action_high)
        else:
            # Discrete action - convert to int
            if hasattr(action, 'item'):  # PyTorch tensor
                action = action.item()
            action = int(action)
        
        # Step environment
        result = self.env.step(action)
        
        # Gymnasium returns (obs, reward, terminated, truncated, info)
        # Old gym returns (obs, reward, done, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs, reward, done, info = result
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} values")
        
        # Update tracking
        self.episode_step += 1
        self.episode_reward += reward
        
        # Get visual frame
        frame = self._get_frame()
        
        return {'frame': frame, 'obs': obs}, reward, done, info
    
    def _get_frame(self) -> np.ndarray:
        """Get current visual frame from environment"""
        default_frame_shape = (600, 600, 3)

        try:
            # Gymnasium/Gym API: render() returns the frame directly
            frame = self.env.render()
            
            if frame is None:
                logger.debug("Render returned None, using blank frame")
                return np.zeros(default_frame_shape, dtype=np.uint8)
            
            # Ensure correct format
            if isinstance(frame, np.ndarray):
                # Ensure uint8
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # Ensure 3 channels (RGB)
                if len(frame.shape) == 2:
                    # Grayscale -> RGB
                    frame = np.stack([frame] * 3, axis=-1)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 1:
                        # Single channel -> RGB
                        frame = np.repeat(frame, 3, axis=-1)
                    elif frame.shape[2] == 4:
                        # RGBA -> RGB (remove alpha)
                        frame = frame[:, :, :3]
                    elif frame.shape[2] != 3:
                        logger.warning(f"Unexpected frame shape: {frame.shape}")
                        return np.zeros(default_frame_shape, dtype=np.uint8)
                
                return frame
            
            else:
                logger.warning(f"Render returned non-array: {type(frame)}")
                return np.zeros(default_frame_shape, dtype=np.uint8)
            
        except Exception as e:
            logger.debug(f"Error getting frame: {e}")
            # Return blank frame on any error
            return np.zeros(default_frame_shape, dtype=np.uint8)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics"""
        return {
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_step,
            'episode_duration': time.time() - self.episode_start_time
        }
    
    def close(self):
        """Close environment"""
        if hasattr(self, 'env'):
            self.env.close()
        
        # Close virtual display if we created one
        if self.virtual_display:
            try:
                self.virtual_display.stop()
                logger.info("Virtual display stopped")
            except Exception:
                pass
        
        logger.info(f"Environment {self.env_name} closed")


class HighwayEnvWrapper(MultiModalEnvWrapper):
    """Specialized wrapper for highway-env"""
    
    def __init__(self, 
                 scenario: str = "highway-fast-v0",
                 config: Optional[Dict] = None):
        
        # Check if this is actually a highway-env scenario
        is_highway = "highway" in scenario.lower()
        
        if is_highway:
            # Default highway-env config
            default_config = {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": False,
                    "normalize": True
                },
                "action": {
                    "type": "DiscreteMetaAction"
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "duration": 40,  # seconds
                "simulation_frequency": 15,  # Hz
                "policy_frequency": 5,  # Hz
                "reward_speed_range": [20, 30],
                "collision_reward": -1.0,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "normalize_reward": True
            }
            
            if config:
                default_config.update(config)
            
            super().__init__(
                env_name=scenario,
                env_config=default_config,
                render_mode="rgb_array"
            )
        else:
            # For non-highway envs, pass config as-is (or None)
            super().__init__(
                env_name=scenario,
                env_config=config,
                render_mode="rgb_array"
            )
        
        logger.info(f"Highway/Env wrapper initialized: {scenario}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset and return specialized observation dict"""
        result = self.env.reset()
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        
        frame = self._get_frame()
        
        return {'frame': frame, 'kinematics': obs}

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Step and return specialized observation dict"""
        
        # Action conversion from base class
        if self.continuous_actions:
            if isinstance(action, (int, float)): action = np.array([action])
            elif hasattr(action, 'cpu'): action = action.cpu().numpy()
            action = np.clip(action, self.action_low, self.action_high)
        else:
            if hasattr(action, 'item'): action = action.item()
            action = int(action)

        result = self.env.step(action)
        
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        self.episode_step += 1
        self.episode_reward += reward
        
        frame = self._get_frame()
        
        return {'frame': frame, 'kinematics': obs}, reward, done, info
    
    def _get_frame(self) -> np.ndarray:
        """Override to ensure proper frame extraction from highway-env"""
        default_shape = (600, 600, 3)

        try:
            # Highway-env specific rendering
            frame = self.env.render()
            
            if frame is None:
                # Force render with mode
                if hasattr(self.env, 'render_mode'):
                    self.env.render_mode = "rgb_array"
                frame = self.env.render()
            
            if frame is not None:
                # Ensure correct format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                
                # Ensure RGB (highway-env sometimes returns RGBA)
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                return frame
            else:
                logger.warning("Highway-env render returned None")
                return np.zeros(default_shape, dtype=np.uint8)
                
        except Exception as e:
            logger.error(f"Error rendering highway-env: {e}")
            return np.zeros(default_shape, dtype=np.uint8)
