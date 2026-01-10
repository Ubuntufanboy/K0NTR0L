import json
from pathlib import Path
import numpy as np

path = Path("../out/ng/0001_ACTIONS.json")
if not path.exists():
    print("File not found")
    exit()

actions = []
with open(path, "r") as f:
    for line in f:
        actions.append(json.loads(line))

print(f"Total actions: {len(actions)}")

# Check SOUTH (Jump) and AXIS_LEFTX (Move)
south_presses = sum(1 for a in actions if a.get("SOUTH", 0) > 0)
west_presses = sum(1 for a in actions if a.get("WEST", 0) > 0)
right_presses = sum(1 for a in actions if a.get("DPAD_RIGHT", 0) > 0 or (a.get("AXIS_LEFTX") and a["AXIS_LEFTX"][0] > 16384))

print(f"SOUTH (Jump) presses: {south_presses}")
print(f"WEST (Run) presses: {west_presses}")
print(f"Right presses: {right_presses}")

# Look at AXIS_LEFTX values
left_x = [a["AXIS_LEFTX"][0] for a in actions if "AXIS_LEFTX" in a]
if left_x:
    print(f"AXIS_LEFTX: min={min(left_x)}, max={max(left_x)}, avg={np.mean(left_x)}")
    
# Look at first few AXIS_LEFTX
print(f"First 20 AXIS_LEFTX: {left_x[:20]}")
