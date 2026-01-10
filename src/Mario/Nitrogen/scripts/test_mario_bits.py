import gym_super_mario_bros
import time

env = gym_super_mario_bros.make('SuperMarioBros-v3')

bits = {
    1: "Bit 1",
    2: "Bit 2",
    4: "Bit 4",
    8: "Bit 8",
    16: "Bit 16",
    32: "Bit 32",
    64: "Bit 64",
    128: "Bit 128"
}

print("Testing bits...")
for bit, name in bits.items():
    env.reset()
    start_info = None
    end_info = None
    
    # Run for a few frames to settle
    for _ in range(10):
        res = env.step(0)
        start_info = res[-1]
        
    start_x = start_info['x_pos']
    start_y = start_info['y_pos']
    
    # Hold the button for 30 frames
    for _ in range(30):
        res = env.step(bit)
        end_info = res[-1]
        
    end_x = end_info['x_pos']
    end_y = end_info['y_pos']
    
    dx = end_x - start_x
    dy = end_y - start_y
    
    print(f"{name} ({bit}): dx={dx}, dy={dy}")

env.close()
