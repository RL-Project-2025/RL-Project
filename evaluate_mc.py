import torch
import numpy as np

from envs.make_env import make_env
from networks.actor_critic import ActorCriticNet
from stable_baselines3.common.vec_env import VecNormalize

from evaluation.evaluate_random import evaluate_random_policy
from evaluation.kpi_analysis import analyse_kpis
from evaluation.heuristics import evaluate_heuristic, simple_heuristic


def evaluate_a2c(
    model_path: str,
    vecnorm_path: str,
    num_episodes: int = 50,
    device: str = "cpu",
):
    """
    Evaluate a trained A2C policy under un-normalised conditions.
    """

    # Recreate environment
    env = make_env(
        use_wrapper=False,
        use_normalisation=True,
        reward_scaling=True,
    )

    # Load VecNormalize statistics
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load network
    network = ActorCriticNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(device)

    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()

    returns = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            )

            with torch.no_grad():
                dist, _ = network(obs_tensor)
                action = torch.argmax(dist.probs, dim=-1)

            obs, reward, done, _ = env.step([action.item()])
            ep_return += reward[0]
            done = done[0]

        returns.append(ep_return)

    return np.array(returns)

def evaluate_heuristic_policy(
    vecnorm_path: str,
    heuristic_fn,
    num_episodes: int = 50,
):
    """
    Evaluate a heuristic policy under un-normalised conditions.
    """

    env = make_env(
        use_wrapper=False,
        use_normalisation=True,
        reward_scaling=True,
    )

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    results = evaluate_heuristic(
        env,
        heuristic_fn=heuristic_fn,
        episodes=num_episodes,
    )

    return results

def evaluate_random_policy_wrapped(
    vecnorm_path: str,
    num_episodes: int = 50,
):
    """
    Evaluate a random policy under un-normalised conditions.
    """

    env = make_env(
        use_wrapper=False,
        use_normalisation=True,
        reward_scaling=True,
    )

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    results = evaluate_random_policy(
        env,
        episodes=num_episodes,
    )

    return results


if __name__ == "__main__":

    MODEL_PATH = "models/a2c_mc_20251222_020941.pt"
    VECNORM_PATH = "models/a2c_mc_20251222_020941_vecnormalize.pkl"
    N_EVAL_EPISODES = 50

    # --- A2C ---
    a2c_returns = evaluate_a2c(
        model_path=MODEL_PATH,
        vecnorm_path=VECNORM_PATH,
        num_episodes=N_EVAL_EPISODES,
    )

    # --- Random baseline ---
    random_results = evaluate_random_policy_wrapped(
        vecnorm_path=VECNORM_PATH,
        num_episodes=N_EVAL_EPISODES,
    )


    # --- Heuristic baseline ---
    heuristic_results = evaluate_heuristic_policy(
        vecnorm_path=VECNORM_PATH,
        heuristic_fn=simple_heuristic,
        num_episodes=N_EVAL_EPISODES,
    )
    

    # --- Print summary ---
    print("\nA2C:")
    print(f"Mean: {a2c_returns.mean():.2f}, Std: {a2c_returns.std():.2f}")

    print("\nRandom:")
    print(
        f"Mean: {random_results['mean_reward']:.2f}, "
        f"Std: {random_results['std_reward']:.2f}"
    )


    print("\nHeuristic:")
    print(
        f"Mean: {heuristic_results['mean_reward']:.2f}, "
        f"Std: {heuristic_results['std_reward']:.2f}"
    )

    # --- KPI analysis (extend if desired) ---
    analyse_kpis(
        a2c_returns,
        random_results["all_rewards"],
        heuristic_results["all_rewards"],
    )
