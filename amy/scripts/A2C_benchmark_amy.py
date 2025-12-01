import gymnasium as gym
from gym4real.envs.wds.utils import parameter_generator
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy


# ************* Utility funcs / classes
def create_env():
    params = parameter_generator()
    env = gym.make('gym4real/wds-v0', **{'settings': params})
    obs,info = env.reset()
    done = False
    return env

# ************* Train the model 
env = create_env()
NUM_TIMESTEPS = 100000
MODEL_FILE_NAME = "A2C_model_" + str(NUM_TIMESTEPS) + "_ts"
A2C_model = A2C("MlpPolicy", env, verbose=1) #device = "mps"
A2C_model.learn(total_timesteps=NUM_TIMESTEPS)
A2C_model.save(MODEL_FILE_NAME)

# ************* Load the model as a test
del A2C_model #delete current model to test if loading works 
A2C_model = A2C.load(MODEL_FILE_NAME, env=env)

# ************* Evaluate the model
NUM_EVAL_EP = 20
# evaluate_policy() runs the policy for n_eval_episodes episodes and outputs the average and std return per episode
mean_reward, std_reward = evaluate_policy(A2C_model, A2C_model.get_env(), n_eval_episodes=NUM_EVAL_EP)
print(mean_reward, std_reward)