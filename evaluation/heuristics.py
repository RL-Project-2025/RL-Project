def simple_heuristic(state):
    # Turn pump ON if tank level < 50%
    tank_level = state[0]  
    if tank_level < 0.5:
        return 1
    return 0

def evaluate_heuristic(env, heuristic_fn, episodes=5):
    rewards = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_r = 0

        while not done:
            action = heuristic_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += reward

        rewards.append(total_r)

    return {
        "mean_reward": sum(rewards)/len(rewards),
        "all_rewards": rewards
    }
