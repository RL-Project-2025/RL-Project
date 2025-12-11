#!/usr/bin/env python3

import os
import sys
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import gym4real
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import RecurrentPPO
from gym4real.envs.wds.utils import parameter_generator

MODELS_DIR = "../models"

if not os.path.exists(MODELS_DIR):
    print("Directory ", MODELS_DIR, "not found") #Why no printf? cause im dumb.
    sys.exit(1)

files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
files.sort()

if not files:
    print("No models found, god bless your soul")

print(f"Select model to evaluate:")
for i, f in enumerate(files):
    print(f"{i+1}. {f}")

try:
    choice = int(input("Enter num: ")) - 1
    if choice < 0 or choice >= len(files): raise ValueError #Man do I love oneliners
except:
    print("Invalid selection")
    sys.exit(1)

filename = files[choice]
model_path = os.path.join(MODELS_DIR, filename)

if os.path.exists("../gym4ReaL"):
    os.chdir("../gym4ReaL")
elif os.path.exists("gym4ReaL"):
    os.chdir("gym4ReaL")

params = parameter_generator(
    hydraulic_step=3600,
    duration=604800,
    seed=1234,
    world_options="gym4real/envs/wds/world_anytown.yaml"
)

env = gym.make("gym4real/wds-v0", settings=params)
name = filename.lower()

#Bout to see peak productivity
if "dqn" in name:
    model_cls = DQN
elif "recurrent" in name:
    model_cls = RecurrentPPO
elif "a2c" in name:
    model_cls = A2C
else:
    model_cls = PPO

print(f"Loading {filename} as {model_cls.__name__}...")

model = model_cls.load(model_path, env=env)

obs, _ = env.reset()
rewards = []
done = False

print("Running evaluation...")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    rewards.append(reward)
    done = terminated or truncated

print(f"Total Reward: {sum(rewards):.2f}")

plt.figure(figsize=(12, 6))
plt.plot(rewards, alpha=0.6, label="Step Reward")
plt.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f"Mean: {np.mean(rewards):.2f}")
plt.title(f"Evaluation: {filename}\nTotal: {sum(rewards):.2f}")
plt.xlabel("Steps (Hours)")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

output_name = f"../graph_{filename.replace('.zip', '')}.png"
plt.savefig(output_name)
print(f"Graph saved to {output_name}")
