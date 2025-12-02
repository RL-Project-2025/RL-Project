def collect_reward_components(model, env):
    obs, info = env.reset()
    done = False

    components = {
        "pressure": [],
        "leak": [],
        "demand_deficit": [],
        "energy": []
    }

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        pure = info["pure_rewards"]
        components["pressure"].append(pure["pressure_reward"])
        components["leak"].append(pure["leak_reward"])
        components["demand_deficit"].append(pure["demand_deficit_reward"])
        components["energy"].append(pure["energy_reward"])

    return components
