import os
import sys
import time
import logging
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from policy import PolicyNetwork
from encoders import MultiModalEncoder
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mario_imitation_policy.pt")
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
GAME_TIME_LIMIT = 5 * 60  # 5 minutes in seconds

class GymAdapter(gym.Env):
    """Adapter to make old Gym envs compatible with Gymnasium"""
    def __init__(self, old_env):
        self.env = old_env
        # Convert spaces? Usually they are compatible enough for basic usage
        self.action_space = old_env.action_space
        self.observation_space = old_env.observation_space
        self.render_mode = "rgb_array"
        self.metadata = getattr(old_env, "metadata", {"render_modes": ["rgb_array"]})
        
    def reset(self, seed=None, options=None):
        # Handle seeding if possible, otherwise ignore
        obs = self.env.reset()
        return obs, {}
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info
    
    def render(self):
        if hasattr(self.env.unwrapped, 'screen'):
            return self.env.unwrapped.screen.copy()
        return self.env.render(mode='rgb_array')
        
    def close(self):
        self.env.close()

def map_action(action_tensor):
    """
    Map continuous 21-dim action vector to discrete NES action index (COMPLEX_MOVEMENT)
    """
    # 0: WEST (B) -> Sprint/Fire
    # 1: SOUTH (A) -> Jump
    # 3: DPAD_DOWN
    # 4: DPAD_LEFT
    # 5: DPAD_RIGHT
    # 6: DPAD_UP
    # 8: AXIS_LEFTX (Left/Right analog)
    
    if hasattr(action_tensor, 'cpu'):
        action = action_tensor.squeeze().cpu().numpy()
    else:
        action = action_tensor
    
    # Thresholds
    pressed_threshold = 0.5
    axis_threshold = 0.3
    
    # Determine intended buttons
    target_set = set()
    
    # B (Sprint/Fire)
    if action[0] > pressed_threshold:
        target_set.add('B')
        
    # A (Jump)
    if action[1] > pressed_threshold:
        target_set.add('A')
        
    # Directions
    # DPAD
    if action[6] > pressed_threshold: target_set.add('up')
    if action[3] > pressed_threshold: target_set.add('down')
    if action[4] > pressed_threshold: target_set.add('left')
    if action[5] > pressed_threshold: target_set.add('right')
    
    # Analog Stick Override
    if action[8] > axis_threshold: target_set.add('right')
    if action[8] < -axis_threshold: target_set.add('left')
    
    # Resolve conflicts
    if 'left' in target_set and 'right' in target_set:
        target_set.remove('left') # Bias towards right
    
    if 'up' in target_set and 'down' in target_set:
        target_set.remove('up')
        target_set.remove('down')
        
    # Find best match in COMPLEX_MOVEMENT
    best_score = -999
    best_idx = 0
    
    for i, move in enumerate(COMPLEX_MOVEMENT):
        move_set = set(move)
        if 'NOOP' in move_set: move_set = set()
        
        # Intersection: Features we wanted and got
        intersection = len(target_set.intersection(move_set))
        
        # Missing: Features we wanted but didn't get
        missing = len(target_set - move_set)
        
        # Extra: Features we didn't want but got
        extra = len(move_set - target_set)
        
        # Weighted score: Prioritize getting what we want, penalize extra buttons lightly
        score = intersection * 2 - missing * 1 - extra * 0.5
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx

def main():
    # Setup virtual display
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1400, 900))
        display.start()
        logger.info("Virtual display started.")
    except ImportError:
        logger.warning("pyvirtualdisplay not found. If running headless, rendering might fail.")
    except Exception as e:
        logger.warning(f"Failed to start virtual display: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Load Encoders
    logger.info("Loading Encoders...")
    encoders = MultiModalEncoder(kinematics_dim=1, device=device)
    
    # 2. Load Policy
    logger.info(f"Loading Model from {MODEL_PATH}...")
    action_dim = 21
    policy = PolicyNetwork(
        input_dim=0,
        action_dim=action_dim,
        text_dim=encoders.text_dim,
        vision_dim=encoders.vision_dim,
        sensor_dim=0,
        continuous=True
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        policy.load_state_dict(state_dict)
        logger.info("Model loaded successfully.")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")
        return

    policy.eval()
    
    # 3. Setup Environment
    env_id = 'SuperMarioBros-v0'
    logger.info(f"Setting up environment: {env_id}")
    
    # Create raw env directly to bypass gym compatibility issues
    try:
        raw_env = SuperMarioBrosEnv()
    except Exception as e:
        logger.error(f"Failed to instantiate SuperMarioBrosEnv directly: {e}")
        # Fallback to make but might fail
        raw_env = gym_super_mario_bros.make(env_id)
        
    raw_env = JoypadSpace(raw_env, COMPLEX_MOVEMENT)
    
    # Adapt to Gymnasium
    env = GymAdapter(raw_env)
    
    # Wrap with RecordVideo (Gymnasium)
    env = RecordVideo(env, VIDEO_DIR, name_prefix="mario_play", episode_trigger=lambda x: True)
    
    # 4. Play Loop
    logger.info("Starting gameplay...")
    
    obs, _ = env.reset()
    done = False
    total_steps = 0
    start_time = time.time()
    last_log_time = 0
    
    # Instruction
    instruction = "Run right and jump over obstacles"
    logger.info(f"Instruction: {instruction}")
    
    # Encode instruction once
    text_emb = encoders.text_encoder.encode(instruction).to(device)
    
    # Hidden state for GRU
    hidden = None
    
    try:
        while True:
            # Process Observation
            # obs is (240, 256, 3) numpy array
            
            # Encode frame
            vision_emb = encoders.vision_encoder.encode(obs).to(device)
            
            # Prepare input
            # Text + Vision
            # Shape: (1, Dim) -> GRU expects (Batch, Seq, Dim) or (Batch, Dim)
            # Policy forward unsqueezes if 2D.
            
            # We need to concatenate [Text | Vision]
            # text_emb is 1D, vision_emb is 1D
            
            combined_input = torch.cat([text_emb, vision_emb], dim=-1).unsqueeze(0) # (1, TotalDim)
            
            # Get action
            with torch.no_grad():
                # Policy returns mean, log_std, hidden (if continuous)
                # We used continuous=True
                mean, _, hidden = policy(combined_input, hidden)
                
            # Map action
            action_idx = map_action(mean)
            
            # Step Env
            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            
            total_steps += 1
            
            # Check time limit (Game time or Real time?)
            # Prompt: "5 minutes game time".
            # Info dict might contain 'time' (left).
            # Usually Mario levels have 400 seconds.
            # "5 minutes game time" probably means keep playing episodes until total duration > 5 mins.
            # But let's check elapsed real time for safety too.
            
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0 and int(elapsed) > 0 and int(elapsed) != last_log_time:
                 logger.info(f"Played {elapsed:.0f} seconds...")
                 last_log_time = int(elapsed)

            if elapsed > GAME_TIME_LIMIT:
                logger.info("Time limit reached.")
                break
                
            if done:
                obs, _ = env.reset()
                hidden = None # Reset hidden state on new episode
                logger.info("Episode finished. Resetting.")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        env.close()
        logger.info(f"Saved video to {VIDEO_DIR}")

if __name__ == "__main__":
    main()
