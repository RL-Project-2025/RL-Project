# INSTRUCTIONS:
# FILE SHOULD BE PLACED IN THE FOLLOWING LOCATION
# gym4ReaL > gym4real > envs > wds >
# TITLE MUST BE 'reward_scaling_wrapper.py'
# This wrapper exists to ensure that rewards are scaled in accordance with timestep length

# PLEASE ENSURE TO ALSO YOU HAVE THE 'envs' folder at the root of your directory with the 'make_env.py' file


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
