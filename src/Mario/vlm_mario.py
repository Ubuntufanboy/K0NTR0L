import os
import time
import base64
import cv2
import numpy as np
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from pyvirtualdisplay import Display
from collections import deque
from PIL import Image
import io
import shutil
import logging
import sys
import torch

# Add Nitrogen to path
sys.path.append(os.path.join(os.path.dirname(__file__), "Nitrogen"))
from nitrogen.inference_session import InferenceSession, load_model

LOG_FILE = "vlm_mario.log"
FAIL_DIR = "fails"
VIDEO_OUTPUT = "mario_replay.mp4"
NITROGEN_CKPT = "Nitrogen/ng.pt"
CONTROL_HZ = 4
FPS = 60
FRAMES_PER_DECISION = FPS // CONTROL_HZ
REWIND_SECONDS = 3
REWIND_STEPS = REWIND_SECONDS * CONTROL_HZ
MAX_RETRIES = 100 
MACRO_LOOKUP = {
    "Spam Jump": ["A", "NOOP"] * 8, 
    "Spam Fire": ["B", "NOOP"] * 8,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Action Mapping for COMPLEX_MOVEMENT
# ['NOOP', 'Right', 'Right + A', 'Right + B', 'Right + A + B', 'A', 'Left', 'Left + A', 'Left + B', 'Left + A + B', 'Down', 'Up']
ACTION_NAMES = [
    "NOOP",
    "Right",
    "Right + Jump",
    "Right + Run",
    "Right + Run + Jump",
    "Jump",
    "Left",
    "Left + Jump",
    "Left + Run",
    "Left + Run + Jump",
    "Down",
    "Up"
]

def load_nitrogen_session():
    """Loads the Nitrogen model and returns the session."""
    logger.info("Loading NitroGen model...")
    ckpt_path = os.path.join(os.path.dirname(__file__), NITROGEN_CKPT)
    
    # Custom loading to bypass input()
    model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(ckpt_path)
    
    selected_game = None
    if game_mapping:
        # Try to find Super Mario Bros
        candidates = [g for g in game_mapping.keys() if "Mario" in g and "Bros" in g]
        if candidates:
            selected_game = candidates[0]
            logger.info(f"Selected game for NitroGen: {selected_game}")
        else:
            logger.info("Super Mario Bros not found in game mapping. Using unconditional.")
    
    session = InferenceSession(
        model,
        ckpt_path,
        tokenizer,
        img_proc,
        ckpt_config,
        game_mapping,
        selected_game,
        old_layout=False, # Default
        cfg_scale=1.0,
        action_downsample_ratio=action_downsample_ratio,
        context_length=None # Default
    )
    return session

# Initialize Nitrogen Session Global
try:
    nitrogen_session = load_nitrogen_session()
except Exception as e:
    logger.error(f"Failed to load NitroGen: {e}")
    sys.exit(1)

def get_vlm_action(frame_rgb, prev_frame_rgb, previous_action_name="None", avoid_actions=None):
    """Queries NitroGen for the next action."""
    
    # NitroGen expects RGB frame
    # Predict returns: {'j_left': array([-0.01, 0.02]), 'j_right': ..., 'buttons': array([0., 0., ...])}
    
    try:
        # Run inference
        # Note: Nitrogen might expect specific resolution, but img_proc handles resizing usually.
        # But we pass numpy array (H, W, C).
        
        preds = nitrogen_session.predict(frame_rgb)
        
        j_left = preds['j_left']   # [horizon, 2] or [2]
        buttons = preds['buttons'] # [horizon, N] or [N]

        # Use the first action in the sequence if multiple are returned
        if j_left.ndim > 1:
            j_left = j_left[0]
        if buttons.ndim > 1:
            buttons = buttons[0]
        
        # Map to NES
        # Stick X: < -0.3 Left, > 0.3 Right
        # Stick Y: < -0.3 Up, > 0.3 Down (Usually Y is inverted? -1 is Up in some pads, +1 in others. Let's assume -1=Up)
        
        stick_x = j_left[0]
        stick_y = j_left[1]
        
        is_left = stick_x < -0.3
        is_right = stick_x > 0.3
        is_down = stick_y > 0.3
        is_up = stick_y < -0.3
        
        # Buttons: Assume 0=A (Jump), 1=B (Run/Fire) (Standard Xbox layout: A=0, B=1)
        # Check button count
        is_jump = False
        is_run = False
        
        if len(buttons) > 0:
            is_jump = buttons[0] > 0.5
        if len(buttons) > 1:
            is_run = buttons[1] > 0.5
            
        # Prioritize movement
        # 0: NOOP
        # 1: Right
        # 2: Right + Jump
        # 3: Right + Run
        # 4: Right + Run + Jump
        # 5: Jump
        # 6: Left
        # 7: Left + Jump
        # 8: Left + Run
        # 9: Left + Run + Jump
        # 10: Down
        # 11: Up
        
        action_idx = 0
        action_name = "NOOP"
        
        if is_right:
            if is_jump and is_run:
                action_idx = 4
                action_name = "Right + Run + Jump"
            elif is_jump:
                action_idx = 2
                action_name = "Right + Jump"
            elif is_run:
                action_idx = 3
                action_name = "Right + Run"
            else:
                action_idx = 1
                action_name = "Right"
        elif is_left:
            if is_jump and is_run:
                action_idx = 9
                action_name = "Left + Run + Jump"
            elif is_jump:
                action_idx = 7
                action_name = "Left + Jump"
            elif is_run:
                action_idx = 8
                action_name = "Left + Run"
            else:
                action_idx = 6
                action_name = "Left"
        elif is_down:
            action_idx = 10
            action_name = "Down"
        elif is_up:
            action_idx = 11
            action_name = "Up"
        elif is_jump:
            action_idx = 5
            action_name = "Jump"
        
        # Log raw for debugging
        logger.info(f"Nitrogen Raw: Lx={stick_x:.2f}, Ly={stick_y:.2f}, Btn={buttons[:4]} -> {action_name}")
        
        return action_idx, action_name

    except Exception as e:
        logger.error(f"NitroGen Inference Failed: {e}")
        return 0, "NOOP"

def save_failure_frames(frame_buffer, start_idx):
    """Saves frames from the buffer to the failures directory."""
    timestamp = int(time.time())
    fail_path = os.path.join(FAIL_DIR, f"fail_{timestamp}")
    os.makedirs(fail_path, exist_ok=True)
    
    # Save last 3 seconds + 1 second of failure
    save_slice = frame_buffer[max(0, start_idx):]
    
    logger.info(f"Saving {len(save_slice)} failure frames to {fail_path}")
    
    for i, frame in enumerate(save_slice):
        # Verify frame
        if frame is None or frame.size == 0:
            logger.warning(f"Empty frame detected at index {i}")
            continue
        
        # Check for unique pixels (basic check for black/solid screens)
        if np.unique(frame).size < 10:
             logger.warning(f"Frame {i} seems blank or solid color.")

        img_path = os.path.join(fail_path, f"frame_{i:04d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Verify file size
        if os.path.getsize(img_path) == 0:
            logger.error(f"Saved frame {img_path} is 0 bytes!")

def check_frozen(frame_buffer, window=30):
    """Checks if the screen has been frozen for 'window' frames."""
    if len(frame_buffer) < window:
        return False
    
    current = frame_buffer[-1]
    # Check against frame 'window' steps ago
    past = frame_buffer[-window]
    
    # If frames are identical
    if np.array_equal(current, past):
        return True
    return False

def replay_actions(env, actions, seed):
    """Resets environment and replays actions to restore state."""
    logger.info(f"Replaying {len(actions)} actions...")
    env.reset()
    # env.seed(seed) # deprecated in new gym, but might be needed for old gym. 
    # nes-py envs might not strictly respect seed for everything but usually start deterministic.
    
    # We can try to speed up by not rendering or processing extra info?
    # env.step is the bottleneck.
    for i, act in enumerate(actions):
        env.step(act)
        if i % 1000 == 0 and i > 0:
            logger.info(f"Replay progress: {i}/{len(actions)}")
    
    logger.info("Replay complete.")

import signal
import sys

# ... existing imports ...

# Global flag for graceful shutdown
RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    logger.info("Signal received. Shutting down gracefully...")
    RUNNING = False

# ... existing functions ...

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start Virtual Display
    try:
        display = Display(visible=0, size=(256, 240))
        display.start()
        logger.info("Virtual Display started.")
    except Exception as e:
        logger.warning(f"Could not start virtual display (might not be needed if SDL_VIDEODRIVER is dummy): {e}")

    # Initialize Environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Set Seed for Determinism
    SEED = 42
    try:
        env.seed(SEED)
        env.action_space.seed(SEED)
    except:
        pass
    
    env.reset()
    
    # Buffers
    action_history = []
    frame_buffer = []
    
    # Memory of failures: step_count -> set of failed action names
    failure_memory = {}
    
    step_count = 0
    decision_count = 0
    
    # Track consecutive VLM decisions to prevent sticking
    consecutive_vlm_actions = 0
    last_vlm_action = -1
    
    current_action = 1 # Start with Right
    current_action_name = "Right"
    
    obs, _, _, _ = env.step(0) # Init step
    action_history.append(0)
    
    current_lives = 2 # Standard start
    
    logger.info("Starting Game Loop...")
    
    try:
        while RUNNING:
            # 1. Decision Logic (4Hz)
            if step_count % FRAMES_PER_DECISION == 0:
                # Check for known bad actions at this step
                avoid_list = list(failure_memory.get(step_count, []))
                
                # Get previous frame for diff
                prev_frame = frame_buffer[-1] if len(frame_buffer) > 0 else None
                
                # Query VLM
                current_action, current_action_name = get_vlm_action(obs, prev_frame, current_action_name, avoid_actions=avoid_list)

                # Check for stuck loop
                if current_action == last_vlm_action:
                    consecutive_vlm_actions += 1
                else:
                    consecutive_vlm_actions = 1
                    last_vlm_action = current_action
                
                if consecutive_vlm_actions > 25:
                    logger.warning(f"Action {current_action_name} repeated > 25 times. Forcing random action.")
                    # Pick random action different from current
                    candidates = [i for i in range(len(ACTION_NAMES)) if i != current_action]
                    current_action = np.random.choice(candidates)
                    current_action_name = ACTION_NAMES[current_action]
                    
                    # Reset trackers
                    last_vlm_action = current_action
                    consecutive_vlm_actions = 1

                decision_count += 1
            
            # 2. Step Environment
            obs, reward, done, info = env.step(current_action)
            action_history.append(current_action)
            
            # 3. Capture & Store Frame
            frame_buffer.append(obs.copy())
            
            # 4. Verify Integrity (Frozen check)
            if step_count % 60 == 0: # Every second
                if check_frozen(frame_buffer, window=60):
                    logger.warning("Detected frozen screen (1 second unchanged).")

            # 5. Death/Failure Check
            lives = info['life']
            if lives < current_lives or done:
                if lives < current_lives:
                    logger.warning(f"Death detected! Lives: {current_lives} -> {lives}")
                    current_lives = lives
                elif done and info.get('flag_get', False):
                    logger.info("Level Completed!")
                    break
                elif done:
                    logger.warning("Game Over (Done=True without flag).")

                # REWIND LOGIC
                logger.info(f"Rewinding {REWIND_SECONDS} seconds...")
                
                rewind_frames = int(REWIND_SECONDS * FPS)
                target_history_len = max(0, len(action_history) - rewind_frames)
                
                if target_history_len < len(action_history):
                    # Save failure
                    save_failure_frames(frame_buffer, len(frame_buffer) - rewind_frames)
                    
                    # Record failure memory
                    # We blame the last 2 decisions
                    current_decision_idx = (step_count // FRAMES_PER_DECISION) * FRAMES_PER_DECISION
                    blame_steps = [current_decision_idx, current_decision_idx - FRAMES_PER_DECISION]
                    
                    for d_step in blame_steps:
                        if d_step >= 0 and d_step < len(action_history):
                            failed_act_idx = action_history[d_step]
                            failed_act_name = ACTION_NAMES[failed_act_idx]
                            
                            if d_step not in failure_memory:
                                failure_memory[d_step] = set()
                            failure_memory[d_step].add(failed_act_name)
                            logger.info(f"Recorded failure memory at step {d_step}: Avoid {failed_act_name}")

                    # Truncate History
                    action_history = action_history[:target_history_len]
                    
                    # Truncate Frames
                    frame_buffer = frame_buffer[:target_history_len]
                    
                    # Replay
                    replay_actions(env, action_history, SEED)
                    
                    # Update Obs from last step of replay
                    logger.info(f"Replaying {len(action_history)} actions...")
                    obs = env.reset()
                    info = {}
                    
                    if len(action_history) == 0:
                         # If rewound completely to start, take initial step to populate info/obs
                         obs, _, done, info = env.step(0)
                         action_history.append(0)
                    else:
                        for i, act in enumerate(action_history):
                            obs, _, done, info = env.step(act)
                            if i % 1000 == 0 and i > 0:
                                logger.info(f"Replay progress: {i}/{len(action_history)}")
                    
                    # Update local state
                    current_lives = info.get('life', 2)
                    step_count = len(action_history)
                    
                    # Perturb action
                    current_action = 0 
                    current_action_name = "NOOP (Reset)"
                    
                else:
                    logger.error("Cannot rewind (history too short). Aborting.")
                    break

            # Update loop counters
            step_count += 1
            
            # Safety break
            if len(frame_buffer) > FPS * 60 * 10: # 10 minutes limit
                logger.info("Time limit reached.")
                break

            # Update lives tracker
            if not done:
                current_lives = info['life']

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Save Video
        logger.info(f"Saving video with {len(frame_buffer)} frames...")
        if len(frame_buffer) > 0:
            height, width, layers = frame_buffer[0].shape
            out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
            for frame in frame_buffer:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            logger.info(f"Video saved to {VIDEO_OUTPUT}")
        
        env.close()
        if 'display' in locals():
            display.stop()

if __name__ == "__main__":
    main()
