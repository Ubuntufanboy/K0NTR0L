import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
from pathlib import Path
from encoders import MultiModalEncoder

logger = logging.getLogger(__name__)

class MarioDataset(Dataset):
    def __init__(self, runs_dir, run_ids, encoders, seq_len=16, target_fps=15, original_fps=60):
        self.seq_len = seq_len
        self.encoders = encoders
        self.samples = []
        
        # Action keys in order
        self.action_keys = [
            "WEST", "SOUTH", "BACK", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT", "DPAD_UP", 
            "GUIDE", "AXIS_LEFTX", "AXIS_LEFTY", "LEFT_SHOULDER", "LEFT_TRIGGER", 
            "AXIS_RIGHTX", "AXIS_RIGHTY", "LEFT_THUMB", "RIGHT_THUMB", "RIGHT_SHOULDER", 
            "RIGHT_TRIGGER", "START", "EAST", "NORTH"
        ]
        
        self.frame_skip = original_fps // target_fps
        
        for run_id in run_ids:
            self._load_run(runs_dir, run_id)
            
    def _load_run(self, runs_dir, run_id):
        json_path = os.path.join(runs_dir, f"{run_id}_ACTIONS.json")
        video_path = os.path.join(runs_dir, f"{run_id}_CLEAN.mp4")
        
        logger.info(f"Loading run {run_id}...")
        
        # Load Actions
        actions = []
        with open(json_path, 'r') as f:
            for line in f:
                actions.append(json.loads(line))
        
        # Downsample actions
        actions = actions[::self.frame_skip]
        
        # Load Video Frames
        # To avoid RAM explosion, we process video and save frames to temp, or keep in memory if small enough.
        # Given 5 mins at 15fps = 4500 frames. 4500 * 224*224*3 * 2 runs ~ 1.3GB. 
        # RAM is likely fine for 15fps.
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        
        pbar = tqdm(total=len(actions), desc=f"Reading Video {run_id}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % self.frame_skip == 0:
                # Resize to expected input size (e.g., 224 or whatever encoder expects)
                # Encoder expects something around 224-256 usually, will resize later in encoder?
                # VisionEncoder in encoders.py has transform with Resize.
                # Just keep as uint8 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                pbar.update(1)
            
            frame_idx += 1
            
            if len(frames) >= len(actions):
                break
                
        cap.release()
        pbar.close()
        
        # Truncate to match length
        min_len = min(len(frames), len(actions))
        frames = frames[:min_len]
        actions = actions[:min_len]
        
        # Generate Instructions (Hindsight)
        # We need an instruction for each sequence.
        # Strategy: Generate one instruction per 'seq_len' block or sliding window.
        # For simplicity in this dataset, we can pre-calculate instructions.
        
        logger.info(f"Generating instructions for {min_len} steps...")
        instructions = self._generate_instructions(frames, actions)
        
        # Store as sequences
        # We want to be able to sample (state, instruction) -> action
        # Or (seq_states, instruction) -> seq_actions
        
        # Convert actions to tensors
        action_tensors = []
        for a in actions:
            vec = []
            for k in self.action_keys:
                val = a[k]
                if isinstance(val, list):
                    val = val[0]
                
                # Normalize
                if "AXIS" in k:
                    val = val / 32768.0
                elif "TRIGGER" in k:
                    val = val / 255.0
                
                vec.append(float(val))
            action_tensors.append(torch.tensor(vec, dtype=torch.float32))
            
        self.samples.append({
            "frames": frames,
            "actions": action_tensors,
            "instructions": instructions
        })
        
    def _generate_instructions(self, frames, actions):
        """
        Mock VLM instruction generation using heuristics.
        In a real scenario, this would batch call Qwen3-VL-4b.
        """
        instructions = []
        
        for i in range(len(frames)):
            # Heuristic based on action
            # Actions are dicts
            a = actions[i]
            parts = []
            
            # Joystick X
            ax_x = a.get("AXIS_LEFTX", [0])
            if isinstance(ax_x, list): ax_x = ax_x[0]
            if ax_x > 8000: parts.append("run right")
            elif ax_x < -8000: parts.append("run left")
            
            # Jump (South/A)
            if a.get("SOUTH", 0): parts.append("jump")
            
            # Fire/Run (West/B)
            if a.get("WEST", 0): parts.append("sprint")
            
            if not parts:
                prompt = "Wait"
            else:
                prompt = " and ".join(parts)
                prompt = prompt.capitalize()
                
            instructions.append(prompt)
            
        return instructions

    def __len__(self):
        total_seqs = 0
        for s in self.samples:
            total_seqs += max(0, len(s["frames"]) - self.seq_len)
        return total_seqs

    def __getitem__(self, idx):
        # Find which run and offset
        for s in self.samples:
            n_seqs = max(0, len(s["frames"]) - self.seq_len)
            if idx < n_seqs:
                # Found it
                frames = s["frames"][idx : idx + self.seq_len]
                actions = s["actions"][idx : idx + self.seq_len]
                
                # Instruction: Use the one at the start or end? 
                # User says "instruction for what happened in past 3 seconds... -> instruction that would result in such action".
                # So the instruction applies to the upcoming segment or the current segment?
                # Usually: Instruction given at t=0, policy executes t=0..T.
                # So we take instruction at idx.
                instruction = s["instructions"][idx]
                
                # Encode instruction
                # Ideally cache this if strings are repeated
                text_emb = self.encoders.text_encoder.encode(instruction)
                
                # Encode frames (Stack them?)
                # VisionEncoder encodes single frame or batch.
                # We return raw frames? No, we should return tensors.
                # VisionEncoder.encode_batch expects list of numpy arrays.
                vision_embs = self.encoders.vision_encoder.encode_batch(frames)
                
                actions_tensor = torch.stack(actions)
                
                return vision_embs, text_emb, actions_tensor
                
            idx -= n_seqs
            
        raise IndexError("Index out of bounds")

