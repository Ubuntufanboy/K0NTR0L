"""
Command Line Interface for Multi-Modal PPO Toolkit
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

from .training_loop import MultiModalPPOTrainingLoop
from .env_wrapper import MultiModalEnvWrapper, HighwayEnvWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mm_ppo_training.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Modal PPO with Vision-Language Model Reward Shaping',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment settings
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--env', type=str, default='highway-fast-v0',
                          help='Gym environment name')
    env_group.add_argument('--env-type', type=str, choices=['generic', 'highway'], 
                          default='highway',
                          help='Type of environment wrapper to use')
    
    # Task settings
    task_group = parser.add_argument_group('Tasks')
    task_group.add_argument('--tasks', type=str, required=True,
                           help='Path to tasks JSON file')
    task_group.add_argument('--task-switch-freq', type=int, default=5,
                           help='Switch task every N episodes')
    
    # VLM settings
    vlm_group = parser.add_argument_group('Vision-Language Model')
    vlm_group.add_argument('--vllm-host', type=str, default='localhost',
                          help='vLLM server host')
    vlm_group.add_argument('--vllm-port', type=int, default=8000,
                          help='vLLM server port')
    vlm_group.add_argument('--vlm-eval-freq', type=int, default=50,
                          help='Run VLM evaluation every N steps')
    vlm_group.add_argument('--frame-fps', type=int, default=5,
                          help='Frame capture rate for VLM (FPS)')
    
    # Training settings
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--episodes', type=int, default=1000,
                            help='Number of training episodes')
    train_group.add_argument('--rollout-steps', type=int, default=2048,
                            help='Steps per rollout')
    train_group.add_argument('--lr', type=float, default=3e-4,
                            help='Learning rate')
    train_group.add_argument('--gamma', type=float, default=0.99,
                            help='Discount factor')
    train_group.add_argument('--gae-lambda', type=float, default=0.95,
                            help='GAE lambda')
    train_group.add_argument('--clip-epsilon', type=float, default=0.2,
                            help='PPO clip epsilon')
    train_group.add_argument('--n-epochs', type=int, default=10,
                            help='PPO epochs per update')
    train_group.add_argument('--batch-size', type=int, default=64,
                            help='Mini-batch size')
    
    # Logging and saving
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log-dir', type=str, default='./logs',
                          help='Logging directory')
    log_group.add_argument('--save-freq', type=int, default=100,
                          help='Save checkpoint every N episodes')
    log_group.add_argument('--checkpoint', type=str, default=None,
                          help='Resume from checkpoint')
    
    # System settings
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument('--device', type=str, default='cuda',
                          choices=['cuda', 'cpu'],
                          help='Device to use')
    sys_group.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval'],
                       help='Mode: train or eval')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Multi-Modal PPO Training")
    logger.info("="*80)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info("="*80)
    
    # Create environment wrapper
    if args.env_type == 'highway':
        env = HighwayEnvWrapper(scenario=args.env)
    else:
        env = MultiModalEnvWrapper(env_name=args.env)
    
    logger.info(f"Environment created: {args.env}")
    
    # Create training loop
    training_loop = MultiModalPPOTrainingLoop(
        env_wrapper=env,
        tasks_path=args.tasks,
        log_dir=args.log_dir,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        device=args.device,
        rollout_steps=args.rollout_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        task_switch_frequency=args.task_switch_freq,
        vlm_eval_frequency=args.vlm_eval_freq,
        frame_capture_fps=args.frame_fps,
        save_frequency=args.save_freq
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        training_loop.trainer.load_checkpoint(args.checkpoint)
    
    # Run training or evaluation
    if args.mode == 'train':
        logger.info(f"Starting training for {args.episodes} episodes...")
        training_loop.train(n_episodes=args.episodes)
    else:
        logger.info(f"Starting evaluation for {args.eval_episodes} episodes...")
        training_loop.evaluate(n_episodes=args.eval_episodes, render=True)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
