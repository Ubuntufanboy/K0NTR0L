import os
import sys
import time
import json
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

# Super Mario Bros and Headless rendering imports
try:
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from pyvirtualdisplay import Display
except ImportError:
    print("Missing dependencies. Please ensure gym-super-mario-bros, nes-py, and pyvirtualdisplay are installed.")
    sys.exit(1)

from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="VLM Inference on Super Mario Bros")
parser.add_argument("--process", type=str, default="SuperMarioBros-v3", help="Game to play (e.g. SuperMarioBros-v3)")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--duration", type=int, default=300, help="Duration in seconds (default 5 minutes)")
parser.add_argument("--no-viz", action="store_true", help="Disable visualization and recording for faster execution")

args = parser.parse_args()

# Start virtual display for headless Linux
print("Starting virtual display...")
display = Display(visible=0, size=(1024, 768))
display.start()

try:
    policy = ModelClient(port=args.port)
    policy.reset()
    policy_info = policy.info()
except Exception as e:
    print(f"Failed to connect to model server: {e}")
    display.stop()
    sys.exit(1)

action_downsample_ratio = policy_info["action_downsample_ratio"]

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

PATH_DEBUG = PATH_REPO / "debug"
PATH_DEBUG.mkdir(parents=True, exist_ok=True)

PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = 0.5
JOYSTICK_THRESHOLD = 8000 # Lowered from 16384 for better responsiveness

# Find next video number
video_files = sorted(PATH_OUT.glob("*_DEBUG.mp4"))
if video_files:
    existing_numbers = [f.name.split("_")[0] for f in video_files]
    existing_numbers = [int(n) for n in existing_numbers if n.isdigit()]
    next_number = max(existing_numbers) + 1
else:
    next_number = 1

PATH_MP4_DEBUG = PATH_OUT / f"{next_number:04d}_DEBUG.mp4"
PATH_MP4_CLEAN = PATH_OUT / f"{next_number:04d}_CLEAN.mp4"
PATH_ACTIONS = PATH_OUT / f"{next_number:04d}_ACTIONS.json"

def preprocess_img(main_image):
    # Resize to 256x256 as expected by the model
    return main_image.resize((256, 256), Image.Resampling.NEAREST)

zero_action = OrderedDict(
        [
            ("WEST", 0),
            ("SOUTH", 0),
            ("BACK", 0),
            ("DPAD_DOWN", 0),
            ("DPAD_LEFT", 0),
            ("DPAD_RIGHT", 0),
            ("DPAD_UP", 0),
            ("GUIDE", 0),
            ("AXIS_LEFTX", np.array([0], dtype=np.int64)),
            ("AXIS_LEFTY", np.array([0], dtype=np.int64)),
            ("LEFT_SHOULDER", 0),
            ("LEFT_TRIGGER", np.array([0], dtype=np.int64)),
            ("AXIS_RIGHTX", np.array([0], dtype=np.int64)),
            ("AXIS_RIGHTY", np.array([0], dtype=np.int64)),
            ("LEFT_THUMB", 0),
            ("RIGHT_THUMB", 0),
            ("RIGHT_SHOULDER", 0),
            ("RIGHT_TRIGGER", np.array([0], dtype=np.int64)),
            ("START", 0),
            ("EAST", 0),
            ("NORTH", 0),
        ]
    )

TOKEN_SET = BUTTON_ACTION_TOKENS

class MarioEnvShim:
    """A shim to make SuperMarioBros look like the GamepadEnv used in the original script."""
    def __init__(self, game_id='SuperMarioBros-v3'):
        print(f"Initializing {game_id}...")
        self.env = gym_super_mario_bros.make(game_id)
        # NES buttons in nes-py: ['right', 'left', 'down', 'up', 'start', 'select', 'B', 'A']
        
    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple): # Gymnasium style compatibility
            obs = obs[0]
        return Image.fromarray(obs)

    def step(self, action):
        # Convert GamepadEnv action dict to Mario bitmask
        # NES Mapping: Right(128), Left(64), Down(32), Up(16), Start(8), Select(4), B(2), A(1)
        mario_action = 0
        # Right
        if action.get("DPAD_RIGHT") or (isinstance(action.get("AXIS_LEFTX"), (np.ndarray, list)) and action["AXIS_LEFTX"][0] > JOYSTICK_THRESHOLD):
            mario_action |= 128
        # Left
        if action.get("DPAD_LEFT") or (isinstance(action.get("AXIS_LEFTX"), (np.ndarray, list)) and action["AXIS_LEFTX"][0] < -JOYSTICK_THRESHOLD):
            mario_action |= 64
        # Down
        if action.get("DPAD_DOWN") or (isinstance(action.get("AXIS_LEFTY"), (np.ndarray, list)) and action["AXIS_LEFTY"][0] > JOYSTICK_THRESHOLD):
            mario_action |= 32
        # Up
        if action.get("DPAD_UP") or (isinstance(action.get("AXIS_LEFTY"), (np.ndarray, list)) and action["AXIS_LEFTY"][0] < -JOYSTICK_THRESHOLD):
            mario_action |= 16
        # Start
        if action.get("START"):
            mario_action |= 8
        # Select
        if action.get("BACK"):
            mario_action |= 4
        # B (Run)
        if action.get("WEST"):
            mario_action |= 2
        # A (Jump)
        if action.get("SOUTH"):
            mario_action |= 1
            
        res = self.env.step(mario_action)
        if len(res) == 4:
            obs, reward, done, info = res
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = res
            
        return Image.fromarray(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()
    
    def pause(self): pass
    def unpause(self): pass

print("Model loaded, starting environment...")
env = MarioEnvShim(game_id=args.process)

obs = env.reset()

frames_count = 0
max_frames = args.duration * 60 # e.g., 5 minutes at 60 FPS
step_count = 0

if not args.no_viz:
    print(f"Recording to {PATH_MP4_DEBUG}")
    debug_recorder = VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="ultrafast")
    clean_recorder = VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="ultrafast")
else:
    debug_recorder = None
    clean_recorder = None

try:
    while frames_count < max_frames:
        obs_pre = preprocess_img(obs)
        
        try:
            pred = policy.predict(obs_pre)
        except Exception as e:
            print(f"Error during policy prediction: {e}")
            break

        j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]

        n = len(buttons)
        assert n == len(j_left) == len(j_right), "Mismatch in action lengths"

        env_actions = []

        for i in range(n):
            move_action = zero_action.copy()

            xl, yl = j_left[i]
            xr, yr = j_right[i]
            move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.int64)
            move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.int64)
            move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.int64)
            move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.int64)
            
            button_vector = buttons[i]
            assert len(button_vector) == len(TOKEN_SET), "Button vector length does not match token set length"
            
            for name, value in zip(TOKEN_SET, button_vector):
                if "TRIGGER" in name:
                    move_action[name] =  np.array([value * 255], dtype=np.int64)
                else:
                    move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0

            env_actions.append(move_action)

        print(f"Executing {len(env_actions)} model actions (ratio {action_downsample_ratio}). Progress: {frames_count}/{max_frames} frames")

        batch_terminated = False
        for i, a in enumerate(env_actions):
            if batch_terminated or frames_count >= max_frames:
                break

            if NO_MENU:
                a["GUIDE"] = 0
                a["START"] = 0
                a["BACK"] = 0

            for _ in range(action_downsample_ratio):
                if frames_count >= max_frames:
                    break
                
                obs, reward, terminated, truncated, info = env.step(action=a)
                frames_count += 1

                if terminated or truncated:
                    obs = env.reset()
                    batch_terminated = True
                    break # Break out of downsample loop

                if not args.no_viz:
                    # Recording
                    obs_viz = np.array(obs).copy()
                    # Resize for output videos - Use lower resolution for speed
                    clean_viz = cv2.resize(obs_viz, (1280, 720), interpolation=cv2.INTER_NEAREST)
                    debug_viz = create_viz(
                        cv2.resize(obs_viz, (640, 360), interpolation=cv2.INTER_NEAREST),
                        i,
                        j_left,
                        j_right,
                        buttons,
                        token_set=TOKEN_SET
                    )
                    debug_recorder.add_frame(debug_viz)
                    clean_recorder.add_frame(clean_viz)
            
            if batch_terminated:
                break # Break out of actions loop to get fresh prediction after death

        # Append env_actions to JSONL
        with open(PATH_ACTIONS, "a") as f:
            for i, a in enumerate(env_actions):
                a_copy = a.copy()
                for k, v in a_copy.items():
                    if isinstance(v, np.ndarray):
                        a_copy[k] = v.tolist()
                a_copy["step"] = step_count
                a_copy["substep"] = i
                json.dump(a_copy, f)
                f.write("\n")

        step_count += 1
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    if debug_recorder: debug_recorder.close()
    if clean_recorder: clean_recorder.close()
    env.close()
    display.stop()
    print(f"Done. Saved {frames_count} frames to {PATH_OUT}")