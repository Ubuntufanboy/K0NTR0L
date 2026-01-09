import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_replay_determinism():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Set Seed
    SEED = 42
    try:
        env.seed(SEED)
        env.action_space.seed(SEED)
    except:
        pass
    
    obs_original = env.reset()
    action_history = []
    
    logger.info("Running original sequence...")
    # Generate random actions or fixed actions
    # We use fixed to be sure
    actions = [1, 2, 5, 1, 0, 1, 2, 5] * 10 # 80 steps
    
    for act in actions:
        obs, _, _, _ = env.step(act)
        action_history.append(act)
        
    final_obs_original = obs.copy()
    
    logger.info("Resetting and replaying...")
    env.reset()
    # Note: If env.reset() doesn't fully reset internal emulator state (like RNG), this might fail.
    # But usually SMB is deterministic.
    
    obs_replay = None
    for i, act in enumerate(action_history):
        obs_replay, _, _, _ = env.step(act)
        
    logger.info("Comparing final observations...")
    
    if np.array_equal(final_obs_original, obs_replay):
        logger.info("SUCCESS: Replay produced identical observation.")
    else:
        diff = np.sum(np.abs(final_obs_original.astype(int) - obs_replay.astype(int)))
        logger.error(f"FAILURE: Replay produced different observation. Diff sum: {diff}")
        # Save images for inspection?
        import cv2
        cv2.imwrite("orig.png", cv2.cvtColor(final_obs_original, cv2.COLOR_RGB2BGR))
        cv2.imwrite("replay.png", cv2.cvtColor(obs_replay, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    test_replay_determinism()
