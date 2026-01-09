"""
Multi-Modal PPO with LLM-based Reward Shaping Toolkit
A production-ready framework for training RL agents with vision-language model evaluation
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import io
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TaskDefinition:
    """Schema for task definition JSON"""
    task_id: str
    command: str  # Natural language command for the RL agent
    success_criteria: str  # What constitutes success
    proxy_rewards: List[str]  # List of proxy reward descriptions
    episode_length: int = 1000
    reward_scale: float = 1.0


@dataclass
class EvaluationResult:
    """Result from VLM evaluation"""
    timestamp: float
    task_id: str
    command: str
    frame_indices: List[int]
    success_score: float  # 0-1 continuous
    proxy_scores: Dict[str, float]  # proxy_reward_name -> score
    total_reward: float
    vlm_reasoning: str


class TaskRegistry:
    """Manages task definitions and switching"""
    
    def __init__(self, tasks_path: str):
        self.tasks_path = Path(tasks_path)
        self.tasks: List[TaskDefinition] = []
        self.current_task_idx = 0
        self.load_tasks()
        
    def load_tasks(self):
        """Load tasks from JSON file"""
        logger.info(f"Loading tasks from {self.tasks_path}")
        
        if not self.tasks_path.exists():
            self._create_example_schema()
            raise FileNotFoundError(
                f"Tasks file not found at {self.tasks_path}. "
                f"Example schema created at {self.tasks_path.parent / 'task_schema_example.json'}"
            )
        
        with open(self.tasks_path, 'r') as f:
            data = json.load(f)
            
        self.tasks = [TaskDefinition(**task) for task in data['tasks']]
        logger.info(f"Loaded {len(self.tasks)} tasks")
        
    def _create_example_schema(self):
        """Create example task schema"""
        example = {
            "tasks": [
                {
                    "task_id": "drive_fast",
                    "command": "Drive as fast as possible while staying safe",
                    "success_criteria": "Maintain speed above 100 km/h for at least 10 seconds without collision",
                    "proxy_rewards": [
                        "+1 reward for every km/h above 100 km/h",
                        "+2 reward per km/h/s of acceleration above 0",
                        "-10 reward for any collision or near-miss"
                    ],
                    "episode_length": 1000,
                    "reward_scale": 1.0
                },
                {
                    "task_id": "lane_change",
                    "command": "Change lanes smoothly to the left",
                    "success_criteria": "Complete lane change within 5 seconds with smooth steering",
                    "proxy_rewards": [
                        "+5 reward for successful lane change",
                        "+1 reward for smooth steering (low jerk)",
                        "-5 reward for abrupt movements"
                    ],
                    "episode_length": 500,
                    "reward_scale": 1.0
                }
            ]
        }
        
        schema_path = self.tasks_path.parent / 'task_schema_example.json'
        with open(schema_path, 'w') as f:
            json.dump(example, f, indent=2)
        logger.info(f"Created example schema at {schema_path}")
        
    def get_current_task(self) -> TaskDefinition:
        """Get current task"""
        return self.tasks[self.current_task_idx]
    
    def switch_task(self, strategy: str = 'sequential') -> TaskDefinition:
        """Switch to next task based on strategy"""
        if strategy == 'sequential':
            self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        elif strategy == 'random':
            self.current_task_idx = np.random.randint(0, len(self.tasks))
        
        task = self.get_current_task()
        logger.info(f"Switched to task: {task.task_id} - {task.command}")
        return task


class FrameBuffer:
    """Manages frame capture and buffering for VLM evaluation"""
    
    def __init__(self, fps: int = 5, buffer_size: int = 10):
        self.fps = fps
        self.buffer_size = buffer_size
        self.frame_interval = 1.0 / fps
        self.last_capture_time = 0
        self.frames = deque(maxlen=buffer_size)
        self.frame_count = 0
        
    def should_capture(self, current_time: float) -> bool:
        """Check if we should capture a frame"""
        return (current_time - self.last_capture_time) >= self.frame_interval
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to buffer"""
        self.frames.append({
            'image': frame,
            'timestamp': timestamp,
            'index': self.frame_count
        })
        self.frame_count += 1
        self.last_capture_time = timestamp
        
    def get_recent_frames(self, n: int = 5) -> List[Dict]:
        """Get n most recent frames"""
        return list(self.frames)[-n:]
    
    def clear(self):
        """Clear frame buffer"""
        self.frames.clear()
        self.frame_count = 0


class VLMRewardEvaluator:
    """Evaluates agent performance using vision-language model"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
                 vllm_host: str = "localhost",
                 vllm_port: int = 8000):
        self.model_name = model_name
        self.vllm_url = f"http://{vllm_host}:{vllm_port}/v1/chat/completions"
        self.session = None
        self._init_client()
        
        logger.info(f"Initialized VLM evaluator with {model_name}")
        
    def _init_client(self):
        """Initialize HTTP client for vLLM"""
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode numpy image to base64"""
        import base64
        
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
        
        # Encode to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _build_evaluation_prompt(self, 
                                 task: TaskDefinition,
                                 frames: List[Dict]) -> Tuple[str, List[Dict]]:
        """Build prompt for VLM evaluation"""
        
        prompt = f"""You are an expert evaluator for reinforcement learning agents. 

TASK COMMAND: {task.command}

SUCCESS CRITERIA: {task.success_criteria}

PROXY REWARDS TO EVALUATE:
"""
        for i, proxy in enumerate(task.proxy_rewards, 1):
            prompt += f"{i}. {proxy}\n"
        
        prompt += f"""
You are shown {len(frames)} frames captured at 5 FPS from the agent's environment.

EVALUATION INSTRUCTIONS:
1. Analyze the frames to understand what the agent is doing
2. Evaluate how well the agent satisfies the success criteria (score 0.0 to 1.0)
3. For each proxy reward, assign a score (can be negative, zero, or positive based on the description)
4. Provide brief reasoning for your evaluation

RESPONSE FORMAT (JSON only, no other text):
{{
    "success_score": <float 0.0-1.0>,
    "proxy_scores": {{
        "proxy_1": <float>,
        "proxy_2": <float>,
        ...
    }},
    "reasoning": "<brief explanation>"
}}

EXAMPLE for "Drive fast" task:
{{
    "success_score": 0.8,
    "proxy_scores": {{
        "proxy_1": 15.5,
        "proxy_2": 3.2,
        "proxy_3": 0.0
    }},
    "reasoning": "Agent maintains high speed (115 km/h) with steady acceleration. No collisions detected."
}}

Now evaluate the provided frames:"""

        # Build message with images
        content = [{"type": "text", "text": prompt}]
        
        for frame_data in frames:
            img_b64 = self._encode_image_to_base64(frame_data['image'])
            content.append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })
        
        return prompt, content
    
    def evaluate(self, 
                task: TaskDefinition,
                frames: List[Dict]) -> EvaluationResult:
        """Evaluate frames using VLM"""
        
        start_time = time.time()
        
        try:
            prompt, content = self._build_evaluation_prompt(task, frames)
            
            # Call vLLM API
            response = self.session.post(
                self.vllm_url,
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            vlm_text = result['choices'][0]['message']['content']
            
            # Extract JSON from response
            json_start = vlm_text.find('{')
            json_end = vlm_text.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in VLM response")
            
            json_str = vlm_text[json_start:json_end]
            eval_data = json.loads(json_str)
            
            # Calculate total reward
            total_reward = eval_data['success_score'] * 10  # Scale success score
            for proxy_score in eval_data['proxy_scores'].values():
                total_reward += proxy_score
            
            total_reward *= task.reward_scale
            
            evaluation = EvaluationResult(
                timestamp=start_time,
                task_id=task.task_id,
                command=task.command,
                frame_indices=[f['index'] for f in frames],
                success_score=eval_data['success_score'],
                proxy_scores=eval_data['proxy_scores'],
                total_reward=total_reward,
                vlm_reasoning=eval_data['reasoning']
            )
            
            eval_time = time.time() - start_time
            logger.debug(f"VLM evaluation completed in {eval_time:.2f}s: "
                        f"success={evaluation.success_score:.3f}, "
                        f"reward={evaluation.total_reward:.3f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"VLM evaluation failed: {e}")
            # Return default evaluation on failure
            return EvaluationResult(
                timestamp=start_time,
                task_id=task.task_id,
                command=task.command,
                frame_indices=[f['index'] for f in frames],
                success_score=0.0,
                proxy_scores={},
                total_reward=0.0,
                vlm_reasoning=f"Evaluation failed: {str(e)}"
            )


class MetricsLogger:
    """Comprehensive logging and metrics tracking"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.task_performance = {}
        self.evaluations = []
        
        logger.info(f"Metrics logger initialized at {self.log_dir}")
    
    def log_episode(self, 
                   episode: int,
                   task_id: str,
                   total_reward: float,
                   episode_length: int,
                   vlm_evaluations: List[EvaluationResult]):
        """Log episode metrics"""
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        # TensorBoard logging
        self.writer.add_scalar('Episode/Reward', total_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        self.writer.add_scalar(f'Task/{task_id}/Reward', total_reward, episode)
        
        # VLM evaluation metrics
        if vlm_evaluations:
            avg_success = np.mean([e.success_score for e in vlm_evaluations])
            self.writer.add_scalar(f'Task/{task_id}/SuccessScore', avg_success, episode)
            
            for eval_result in vlm_evaluations:
                self.evaluations.append(eval_result)
        
        # Task-specific tracking
        if task_id not in self.task_performance:
            self.task_performance[task_id] = []
        self.task_performance[task_id].append({
            'episode': episode,
            'reward': total_reward,
            'length': episode_length
        })
        
        logger.info(f"Episode {episode} [{task_id}]: "
                   f"Reward={total_reward:.2f}, Length={episode_length}")
    
    def log_training_metrics(self, 
                            step: int,
                            policy_loss: float,
                            value_loss: float,
                            entropy: float):
        """Log training metrics"""
        
        self.writer.add_scalar('Training/PolicyLoss', policy_loss, step)
        self.writer.add_scalar('Training/ValueLoss', value_loss, step)
        self.writer.add_scalar('Training/Entropy', entropy, step)
    
    def save_checkpoint(self, episode: int, model_state: Dict):
        """Save checkpoint with metrics"""
        
        checkpoint_path = self.log_dir / f'checkpoint_ep{episode}.pt'
        
        torch.save({
            'episode': episode,
            'model_state': model_state,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'task_performance': self.task_performance
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def close(self):
        """Close logger"""
        self.writer.close()
        
        # Save final evaluation log
        eval_log_path = self.log_dir / 'evaluations.json'
        with open(eval_log_path, 'w') as f:
            json.dump([asdict(e) for e in self.evaluations], f, indent=2)
        
        logger.info(f"Saved {len(self.evaluations)} evaluations to {eval_log_path}")
