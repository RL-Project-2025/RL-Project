#!/usr/bin/env python3

import gymnasium as gym

class RewardScalingWrapper(gym.Wrapper):
    def __init__(self, env, reference_dt=3600):
        super().__init__(env)
        self.reference_dt = reference_dt
        self._last_elapsed_time = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_elapsed_time = info.get("elapsed_time", 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_time = info.get("elapsed_time", None)

        if current_time is not None and self._last_elapsed_time is not None:
            dt = current_time - self._last_elapsed_time
        else:
            dt = self.reference_dt  # Fallback

        self._last_elapsed_time = current_time

        # THE KEY FIX
        scaled_reward = reward * (dt / self.reference_dt)

        return obs, scaled_reward, terminated, truncated, info
