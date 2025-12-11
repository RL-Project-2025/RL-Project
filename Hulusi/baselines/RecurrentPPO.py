#!/usr/bin/env python3
import os
import gymnasium as gym
import gym4real
from sb3_contrib import RecurrentPPO
from gym4real.envs.wds.utils import parameter_generator

if os.path.exists("gym4ReaL"):
    os.chdir("gym4ReaL")

params = parameter_generator(
    hydraulic_step=3600,
    duration=604800,
    seed=42,
    world_options="gym4real/envs/wds/world_anytown.yaml"
)

env = gym.make("gym4real/wds-v0", settings=params)
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, device="cpu", tensorboard_log="../logs/")
model.learn(total_timesteps=200000, tb_log_name="recurrent_ppo_200k")
model.save("../models/recurrent_ppo_200k")
print("Done")
