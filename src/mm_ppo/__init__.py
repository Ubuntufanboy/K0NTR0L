"""
Multi-Modal PPO with Vision-Language Model Reward Shaping

A production-ready toolkit for training RL agents with vision-language model evaluation.
Supports any Gym environment with human-defined tasks and reward criteria.

Key Features:
- Multi-modal state encoding (text commands + visual observations)
- VLM-based reward shaping with proxy rewards
- Dynamic task switching for multi-task learning
- Environment-agnostic design
- Comprehensive logging and checkpointing

Example Usage:
    ```python
    from mm_ppo import MultiModalPPOTrainingLoop, HighwayEnvWrapper
    
    # Create environment
    env = HighwayEnvWrapper(scenario='highway-fast-v0')
    
    # Create training loop
    trainer = MultiModalPPOTrainingLoop(
        env_wrapper=env,
        tasks_path='tasks.json',
        log_dir='./logs',
        vllm_host='localhost',
        vllm_port=8000
    )
    
    # Train
    trainer.train(n_episodes=1000)
    ```

Command Line Usage:
    ```bash
    python -m mm_ppo.cli --env highway-fast-v0 --tasks tasks.json --episodes 1000
    ```
"""

__version__ = '0.1.0-alpha'
__author__ = 'Multi-Modal PPO Team'
__license__ = 'MIT'

from .core import (
    TaskRegistry,
    TaskDefinition,
    FrameBuffer,
    VLMRewardEvaluator,
    EvaluationResult,
    MetricsLogger
)

from .encoders import (
    TextEncoder,
    VisionEncoder,
    MultiModalEncoder,
    EncoderWrapper
)

from .policy import (
    PolicyNetwork,
    ValueNetwork,
    ActorCritic,
    RolloutBuffer,
    compute_gae
)

from .trainer import (
    PPOTrainer,
    MultiTaskPPOTrainer
)

from .env_wrapper import (
    MultiModalEnvWrapper,
    HighwayEnvWrapper
)

from .training_loop import (
    MultiModalPPOTrainingLoop
)

__all__ = [
    # Core
    'TaskRegistry',
    'TaskDefinition',
    'FrameBuffer',
    'VLMRewardEvaluator',
    'EvaluationResult',
    'MetricsLogger',
    
    # Encoders
    'TextEncoder',
    'VisionEncoder',
    'MultiModalEncoder',
    'EncoderWrapper',
    
    # Policy
    'PolicyNetwork',
    'ValueNetwork',
    'ActorCritic',
    'RolloutBuffer',
    'compute_gae',
    
    # Trainer
    'PPOTrainer',
    'MultiTaskPPOTrainer',
    
    # Environment
    'MultiModalEnvWrapper',
    'HighwayEnvWrapper',
    
    # Training Loop
    'MultiModalPPOTrainingLoop',
]
