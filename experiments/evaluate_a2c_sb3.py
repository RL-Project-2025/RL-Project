from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np

from envs.make_env import make_env


def evaluate_sb3_a2c(
    model_path: str,
    vecnorm_path: str,
    num_episodes: int = 50,
):
    env = make_env(
        use_wrapper=False,
        use_normalisation=True,
        reward_scaling=True,
    )

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = A2C.load(model_path, env=env)

    returns = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_return += reward[0]
            done = done[0]

        returns.append(ep_return)

    return returns

if __name__ == "__main__":
    MODEL_PATH = "models/sb3_a2c.zip"
    VECNORM_PATH = "models/sb3_a2c_vecnormalize.pkl"
    N_EVAL_EPISODES = 50

    returns = evaluate_sb3_a2c(
        model_path=MODEL_PATH,
        vecnorm_path=VECNORM_PATH,
        num_episodes=N_EVAL_EPISODES,
    )

    returns = np.array(returns)

    print("\nSB3 A2C Evaluation")
    print(f"Episodes: {N_EVAL_EPISODES}")
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {returns.std():.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")