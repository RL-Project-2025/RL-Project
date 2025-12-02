import os
import yaml
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator

# -----------------------
# Locate gym4real package
# -----------------------
package_root = os.path.dirname(gym4real.__file__)

# Path to world config file (raw/original)
world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")
print("Loading world config:", world_file)

# -----------------------
# Load YAML
# -----------------------
with open(world_file, "r") as f:
    world_settings = yaml.safe_load(f)


# -----------------------
# Helper: convert "gym4real/..." → absolute path
# -----------------------
def fix_path(path):
    if isinstance(path, str) and path.startswith("gym4real/"):
        # Replace "gym4real/" with absolute full path
        return os.path.join(package_root, path.replace("gym4real/", ""))
    return path


# Fix demand path
world_settings["demand"]["path"] = fix_path(world_settings["demand"]["path"])

# Fix attackers path (if present)
if "attackers" in world_settings:
    world_settings["attackers"]["path"] = fix_path(world_settings["attackers"]["path"])

# Fix INP file path (critical)
world_settings["inp_file"] = fix_path(world_settings["inp_file"])

# -----------------------
# Write corrected YAML to a temporary file
# -----------------------
fixed_world_file = os.path.join(package_root, "envs", "wds", "world_anytown_fixed.yaml")

with open(fixed_world_file, "w") as f:
    yaml.dump(world_settings, f)

print("Fixed config written to:", fixed_world_file)

# -----------------------
# Now load params using the FIXED yaml
# -----------------------
params = parameter_generator(fixed_world_file)

# -----------------------
# Create environment
# -----------------------
print("Resetting the environment...")
env = gym.make("gym4real/wds-v0", settings=params)

obs, info = env.reset()

print("WDS environment loaded successfully!")
print("Observation sample:", obs)
print("Info keys:", info.keys())

env.close()