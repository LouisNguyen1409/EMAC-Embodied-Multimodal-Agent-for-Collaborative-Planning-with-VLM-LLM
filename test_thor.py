"""Diagnostic: test AlfredThorEnv reset directly (no threads) to see the real error."""
import os
import json
import traceback
from alfworld.env.thor_env import ThorEnv
from alfworld.gen import constants

# Find a traj_data.json to test with
alfworld_data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
data_path = os.path.join(alfworld_data, "json_2.1.1", "train")

task_file = None
for root, dirs, files in os.walk(data_path):
    if "traj_data.json" in files and "movable" not in root and "Sliced" not in root:
        task_file = os.path.join(root, "traj_data.json")
        break

if not task_file:
    print("ERROR: No traj_data.json found in", data_path)
    exit(1)

print(f"Using task file: {task_file}")

# Step 1: Create ThorEnv directly
print("\n--- Creating ThorEnv ---")
try:
    env = ThorEnv(player_screen_height=300, player_screen_width=300)
    print("ThorEnv created OK")
except Exception as e:
    print(f"ThorEnv creation FAILED: {e}")
    traceback.print_exc()
    exit(1)

# Step 2: Load traj_data
with open(task_file, "r") as f:
    traj_data = json.load(f)

scene_num = traj_data["scene"]["scene_num"]
scene_name = "FloorPlan%d" % scene_num
print(f"\n--- Resetting to scene: {scene_name} ---")

try:
    env.reset(scene_name)
    print("Reset OK")
except Exception as e:
    print(f"Reset FAILED: {e}")
    traceback.print_exc()
    exit(1)

# Step 3: Restore scene
print("\n--- Restoring scene ---")
try:
    object_poses = traj_data["scene"]["object_poses"]
    object_toggles = traj_data["scene"]["object_toggles"]
    dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    print("Restore scene OK")
except Exception as e:
    print(f"Restore scene FAILED: {e}")
    traceback.print_exc()
    exit(1)

# Step 4: Init action (what AlfredThorEnv.Thor.reset does after restore_scene)
print("\n--- Init action ---")
try:
    init_action = dict(traj_data["scene"]["init_action"])
    print(f"init_action: {init_action}")
    env.step(init_action)
    print("Init action OK")
except Exception as e:
    print(f"Init action FAILED: {e}")
    traceback.print_exc()

# Step 5: Get frame
print("\n--- Getting frame ---")
try:
    frame = env.last_event.frame
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
except Exception as e:
    print(f"Get frame FAILED: {e}")
    traceback.print_exc()

env.stop()
print("\nAll OK!")
