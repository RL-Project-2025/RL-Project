#!/usr/bin/env python3

import gymnasium as gym

class HourlyDecisionWrapper(gym.Wrapper):
    """
    Converts the semi-Markov EPANET environment (variable Δt)
    into a fixed-step MDP of exactly 1-hour decisions.

    - The wrapped env may take many internal micro-steps.
    - Agent issues one action per hour.
    - Rewards are aggregated over each hour.
    - Episode terminates at 604800 seconds (1 week = 168 hours).
    """

    def __init__(self, env, horizon=604800, hour=3600):
        super().__init__(env)
        self.horizon = horizon           # 1 week = 604800s
        self.hour = hour                 # 1 RL step = 3600s
        self.current_time = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_time = 0.0
        return obs, info

    def step(self, action):
        """
        Run EPANET internal simulation until the next hour boundary
        or until the underlying environment terminates.
        """

        target_time = self.current_time + self.hour
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        while True:
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            elapsed = info.get("elapsed_time", None)
            if elapsed is None:
                raise RuntimeError("Underlying env must report elapsed_time in info dict.")

            # Stop if EPANET ended OR we reached the next hour boundary
            if terminated or truncated or elapsed >= target_time:
                self.current_time = min(elapsed, self.horizon)
                done = terminated or truncated or (self.current_time >= self.horizon)
                return obs, total_reward, done, False, info
