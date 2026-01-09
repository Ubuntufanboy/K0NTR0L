import os
import sys
import cv2
import torch
import numpy as np
import gymnasium as gym
from PIL import Image

# Add Nitrogen to path to ensure we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "Nitrogen"))

try:
    from nitrogen.inference_session import InferenceSession, load_model
    from nitrogen.shared import BUTTON_ACTION_TOKENS
except ImportError as e:
    print(f"Error importing Nitrogen: {e}")
    sys.exit(1)

# Global button mapping
btn_map = {name: i for i, name in enumerate(BUTTON_ACTION_TOKENS)}

# Mario movement names for logging
MARIO_ACTIONS = [
    "NOOP", "RIGHT", "RIGHT+JUMP", "RIGHT+RUN", "RIGHT+JUMP+RUN", "JUMP",
    "LEFT", "LEFT+JUMP", "LEFT+RUN", "LEFT+JUMP+RUN", "DOWN", "UP"
]

# Configuration
NITROGEN_CKPT = "Nitrogen/ng.pt"
RESIZE_SHAPE = (256, 256)
CONTEXT_LENGTH = 1 # Current frame only

def setup_mario_env():
    """Sets up the Super Mario Bros environment with gymnasium and JoypadSpace."""
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    
    # Try to make environment with gymnasium compatibility
    # Note: version 7.2.3 of gym-super-mario-bros is usually for old gym,
    # but gymnasium.make can often wrap it with apply_api_compatibility=True.
    try:
        env = gym.make('SuperMarioBros-v0', render_mode='rgb_array', apply_api_compatibility=True)
    except Exception:
        # Fallback for older versions
        import gym as old_gym
        env = old_gym.make('SuperMarioBros-v0')
    
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env

def initialize_nitrogen():
    """Initializes the NitroGen inference session."""
    ckpt_path = os.path.join(os.path.dirname(__file__), NITROGEN_CKPT)
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: NitroGen checkpoint not found at {ckpt_path}")
        print("Please run download_nitrogen.py first or ensure the model is placed correctly.")
        sys.exit(1)
        
    print(f"Loading NitroGen model from {ckpt_path}...")
    # Load model components using NitroGen's internal load_model
    model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(ckpt_path)
    
    # Identify the game label in mapping (if available)
    selected_game = None
    if game_mapping:
        for game_name in game_mapping.keys():
            if game_name and "Mario" in game_name:
                selected_game = game_name
                print(f"Found matching game in mapping: {selected_game}")
                break
    
    # Create the session with history support
    session = InferenceSession(
        model=model,
        ckpt_path=ckpt_path,
        tokenizer=tokenizer,
        img_proc=img_proc,
        ckpt_config=ckpt_config,
        game_mapping=game_mapping,
        selected_game=selected_game,
        old_layout=False, # Default layout [j_left, j_right, buttons]
        cfg_scale=1.0,
        action_downsample_ratio=action_downsample_ratio,
        context_length=CONTEXT_LENGTH
    )
    
    return session

def map_nitro_to_mario(preds):
    """
    Maps NitroGen predictions to COMPLEX_MOVEMENT indices.
    """
    j_left = preds['j_left']
    buttons = preds['buttons']
    
    # If model returns multiple steps (horizon), we take the first one
    if j_left.ndim > 1: j_left = j_left[0]
    if buttons.ndim > 1: buttons = buttons[0]
    
    lx, ly = j_left
    
    # Map buttons using the names from shared.py
    # SOUTH is typically 'A' (Jump)
    # WEST/EAST is typically 'B' (Run/Fire)
    is_jump = buttons[btn_map['SOUTH']] > 0.5
    is_run = buttons[btn_map['WEST']] > 0.5 or buttons[btn_map['EAST']] > 0.5
    
    is_right = lx > 0.3
    is_left = lx < -0.3
    is_down = ly > 0.3
    is_up = ly < -0.3
    
    # Handle D-Pad as well if model uses it
    if not is_right and buttons[btn_map['DPAD_RIGHT']] > 0.5: is_right = True
    if not is_left and buttons[btn_map['DPAD_LEFT']] > 0.5: is_left = True
    if not is_down and buttons[btn_map['DPAD_DOWN']] > 0.5: is_down = True
    if not is_up and buttons[btn_map['DPAD_UP']] > 0.5: is_up = True

    if is_right:
        if is_jump and is_run: return 4
        if is_jump: return 2
        if is_run: return 3
        return 1
    elif is_left:
        if is_jump and is_run: return 9
        if is_jump: return 7
        if is_run: return 8
        return 6
    elif is_jump:
        return 5
    elif is_down:
        return 10
    elif is_up:
        return 11
    
    return 0

def run_loop():
    # Initialize components
    session = initialize_nitrogen()
    env = setup_mario_env()
    
    print("Starting Mario play loop...")
    reset_res = env.reset()
    obs, info = reset_res if isinstance(reset_res, tuple) and len(reset_res) == 2 else (reset_res, {})
    
    total_reward = 0
    frame_count = 0
    
    try:
        while True:
            # 1. Preprocess: Resize to 256x256 as requested
            obs_resized = cv2.resize(obs, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
            obs_pil = Image.fromarray(obs_resized)
            
            # 2. Predict: Feed current and past frames (handled by session buffer)
            preds = session.predict(obs_pil)
            
            # 3. Act: Map to Mario actions
            action = map_nitro_to_mario(preds)
            
            # 4. Step: Gymnasium step
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                obs, reward, done, truncated, info = step_result
            
            total_reward += reward
            frame_count += 1
            
            # Extraction for logging
            j_left = preds['j_left'][0] if preds['j_left'].ndim > 1 else preds['j_left']
            buttons = preds['buttons'][0] if preds['buttons'].ndim > 1 else preds['buttons']
            lx, ly = j_left
            jump_prob = buttons[btn_map['SOUTH']]
            run_prob = max(buttons[btn_map['WEST']], buttons[btn_map['EAST']])
            
            # Logging
            action_name = MARIO_ACTIONS[action]
            x_pos = info.get('x_pos', 0)
            status = info.get('status', 'unknown')
            time_left = info.get('time', 0)
            
            # Every 10 frames or when an event happens, log state
            if frame_count % 10 == 0 or reward != 0 or done or truncated:
                print(f"F:{frame_count:05d} | Act: {action_name:<15} | R: {reward:5.1f} | TotR: {total_reward:7.1f} | X: {x_pos:4d} | Status: {status} | T: {time_left}")
                print(f"       | Raw: LX:{lx:5.2f}, LY:{ly:5.2f}, Jump:{jump_prob:4.2f}, Run:{run_prob:4.2f}")

            if done or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Final X: {x_pos}, Frame Count: {frame_count}")
                reset_res = env.reset()
                obs, info = reset_res if isinstance(reset_res, tuple) and len(reset_res) == 2 else (reset_res, {})
                session.reset()
                total_reward = 0
                frame_count = 0
                print("-" * 40)
                print("Episode reset.")
                
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        env.close()

if __name__ == "__main__":
    run_loop()
