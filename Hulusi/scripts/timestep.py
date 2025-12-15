import os
if os.path.exists('gym4ReaL'):
    os.chdir('gym4ReaL')
import gymnasium as gym
import gym4real
from gym4real.envs.wds.utils import parameter_generator

params = parameter_generator(
    hydraulic_step=3600,
    duration=604800,
    seed=42,
    world_options='gym4real/envs/wds/world_anytown.yaml'
)
env = gym.make('gym4real/wds-v0', settings=params)

obs, info = env.reset()
last_t = info.get('elapsed_time', 0.0)
steps = 0
done = False
truncated = False
while not (done or truncated):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    steps += 1
    t = info.get('elapsed_time', 0)
    dt = t - last_t
    print(f"Step {steps}: elapsed_time = {t}, Δt = {dt}")
    last_t = t

print(f"\nTotal steps: {steps}")
