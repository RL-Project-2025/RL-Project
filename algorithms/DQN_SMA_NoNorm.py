import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import os
import io
import zipfile
from torch.utils.tensorboard import SummaryWriter

# Following code implements a DQN and DDQN agent from scratch
# DDQN Class uses inheritance from original DQN Class
# Uses epsilon decay, more random and exploratory behaviour initially and eventually follows experience 

class Neural_Network(nn.Module):
    
    # Create Q-Network which takes in states and outputs state-action pairs
    def __init__(self, state_dimension, action_dimension, hidden=128):
        super(Neural_Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dimension))
        
    # Calculate Forward pass prediction by network
    def forward(self, x):
        return self.net(x)

class Replay_Buffer:

    # Initialise buffer to have max size N. Any new experiences will replace old experiences if N > max_size
    def __init__(self, N):
        self.buffer = deque(maxlen=N)

    # Add experience to replay buffer
    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Create random sample batch of experience from replay buffer (size = bath_size)
    def mini_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

class DQN_Implementation:
    
    def __init__(self, env, learning_rate=3e-4, buffer_size=50000, batch_size=64, gamma=0.99, tensorboard_log=None, epsilon_decay_flag=True):
        # Initialise parameters env, lr, batch_size, gamma and tensorboard_logs for metrics
        # Buffer Size of 50000 allows for more episodes to be stored, helps with learning unexpected events
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size= buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.epsilon_decay_flag = epsilon_decay_flag

        # Hyperparameters, using epsilon decay to improve learning over episodes
        # Initially agent acts completely random, as training increases follows optimal policy as experience increases
        self.epsilon = 1.0
        self.end_epsilon = 0.01
        self.decay_rate = 0.01
        self.update_frequency = 1000 
        self.steps_completed = 0

        self.experience_memory = Replay_Buffer(buffer_size)  # Initialise Replay Buffer
        
        self.state_dimension = env.observation_space.shape[0]  # State Space Dimension
        self.action_dimension = env.action_space.n  # Action Space Dimension

        self.policy_network = Neural_Network(self.state_dimension, self.action_dimension)  # Initialise policy action-value network
        self.target_network = Neural_Network(self.state_dimension, self.action_dimension)  # Initialise target action-value network
        
        self.target_network.load_state_dict(self.policy_network.state_dict())  # target network has same initial parmaeters as policy network
        self.target_network.eval() 

        self.optim = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)  # Initialise Adam Optimizer for LR

        # Initialise Logging for Tensorboard
        self.writer = None
        if self.tensorboard_log:
            os.makedirs(self.tensorboard_log, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

    def action(self, state):
        if self.epsilon_decay_flag:
            epsilon_val = self.epsilon
        else:
            epsilon_val = random.random()
        
        if random.random() < epsilon_val:  # Randomly select action if prob < epsilon
            return random.randrange(self.action_dimension)
        else:  # otherwise choose action to maximise q (epsilon greedy)
            with torch.no_grad():  # To improve speed and memory for PyTorch
                q_values = self.policy_network(torch.FloatTensor(state).unsqueeze(0))
                return q_values.argmax().item()

    def update_network(self):
        if len(self.experience_memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.experience_memory.mini_batch(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Calculate Policy Network Q-Values Q(s, a)
        policy_q = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            # Calculate Target Network Q-Values using Bellman Equation; if terminal state then target_q = rewards
            target_q = rewards + (self.gamma * self.target_network(next_states).max(1)[0].unsqueeze(1) * (1 - dones))

        huber_loss = nn.HuberLoss(delta=1.0)(policy_q, target_q)  # We normalise rewards with an upper bound of 1 hence delta of 1.0
        
        # Weights only updated for Policy Network using gradient descent, not for Target Network
        self.optim.zero_grad()
        huber_loss.backward()
        self.optim.step()
        
        return huber_loss.item()

    def learn(self, total_timesteps):
        print(f"Train for {total_timesteps} steps")
        state, _ = self.env.reset()
        episode_reward, episode_count = 0, 0
        
        for step in range(1, total_timesteps + 1):
            self.steps_completed += 1
            
            action = self.action(state)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode_reward += reward
            
            flag = terminated
            # Store experience in Replay Buffer
            self.experience_memory.store_experience(state, action, reward, next_state, flag)
            
            state = next_state

            loss_val = self.update_network()
            # If episode finishes or terminates after special event (overflow)
            if terminated or truncated:
                if self.writer:
                    self.writer.add_scalar("rollout/episode_reward_mean", episode_reward, self.steps_completed)
                    self.writer.add_scalar("train/epsilon", self.epsilon, self.steps_completed)
                
                print(f"Step: {self.steps_completed} | Episode Reward: {episode_reward:.3f} | Epsilon: {self.epsilon:.3f}")
                
                state, _ = self.env.reset()
                episode_reward = 0
                episode_count += 1

                # Epsilon Decay formula - only if epsilon_decay_flag is True
                if self.epsilon_decay_flag:
                    self.epsilon = max(self.end_epsilon, self.epsilon * (1-self.decay_rate))

            # Every 100 steps add the train/loss value to the log
            if loss_val is not None and step % 100 == 0 and self.writer:
                self.writer.add_scalar("train/loss", loss_val, self.steps_completed)

            # Every 1000 steps update the target network
            if self.steps_completed % self.update_frequency == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        print("Training finished")
        if self.writer:
            self.writer.flush()
            self.writer.close()
            
    # Save model and .zip file
    def save(self, path):
        if not path.endswith(".zip"):
            path += ".zip"
            
        print(f"Saving model to {path}")
        
        buffer = io.BytesIO()
        torch.save({
            'model_state_dictionary': self.policy_network.state_dict(),
            'optimizer_state_dictionary': self.optim.state_dict(),
            'epsilon': self.epsilon}, buffer)
        buffer.seek(0)
        
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("policy.pth", buffer.read())
            
        print("Model saved")

class Double_DQN_Implementation(DQN_Implementation):

    # Only change required is update_network method when calculating target_q from Bellman Equation
    def update_network(self):
        if len(self.experience_memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.experience_memory.mini_batch(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Calculate Policy Network Q-Values Q(s, a)
        policy_q = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            # Difference between DQN and DDQN
            # Select best actions from policy network, calculate target_q values with these action from the target network
            best_next_actions = self.policy_network(next_states).argmax(1).unsqueeze(1)
            target_q = rewards + (self.gamma * self.target_network(next_states).gather(1, best_next_actions) * (1 - dones))

        huber_loss = nn.HuberLoss(delta=1.0)(policy_q, target_q)  # We normalise rewards with an upper bound of 1 hence delta of 1.0
        
        # Weights only updated for Policy Network using gradient descent, not for Target Network
        self.optim.zero_grad()
        huber_loss.backward()
        self.optim.step()
        
        return huber_loss.item()

if __name__ == '__main__':
    import os
    import gymnasium as gym
    import gym4real
    from gym4real.envs.wds.utils import parameter_generator
    from gym4real.envs.wds.reward_scaling_wrapper import RewardScalingWrapper
    from Normalise import NormaliseObservation

    if os.path.exists("gym4ReaL"):
        os.chdir("gym4ReaL")
    
    params = parameter_generator(
        hydraulic_step=3600,
        duration=604800,
        seed=42,
        world_options="gym4real/envs/wds/world_anytown.yaml",
    )

    env = gym.make("gym4real/wds-v0", settings=params)
    env = RewardScalingWrapper(env)

    use_ddqn = False
    
    if use_ddqn:
        agent = Double_DQN_Implementation(env, tensorboard_log="../logs/ddqn")
    else:
        agent = DQN_Implementation(env, tensorboard_log="../logs/DQN_SMA_NotNormalised")
    
    agent.learn(total_timesteps=200000)
    agent.save("../models/DQN_SMA_NotNormalised")
    print(f"\nTraining complete. Model saved to ../models/DQN_SMA_NotNormalised.zip")
    print(f"View logs: tensorboard --logdir=../logs")
