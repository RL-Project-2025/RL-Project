from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure

from envs.make_env import make_env


def train_sb3_a2c(
    total_timesteps: int = 300_000,
    log_dir: str = "runs/sb3_a2c",
):
    # Environment (same config as your custom A2C)
    env = make_env(
        use_wrapper=False,
        use_normalisation=True,
        reward_scaling=True,
    )

    # SB3 logger (TensorBoard compatible)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=5,                 # truncated n-step returns
        gae_lambda=1.0,            # MC-style advantage
        ent_coef=0.01,
        vf_coef=0.5,
        normalize_advantage=True,  # IMPORTANT
        verbose=1,
    )

    model.set_logger(logger)

    model.learn(total_timesteps=total_timesteps)

    model.save("models/sb3_a2c")
    env.save("models/sb3_a2c_vecnormalize.pkl")


if __name__ == "__main__":
    train_sb3_a2c()
