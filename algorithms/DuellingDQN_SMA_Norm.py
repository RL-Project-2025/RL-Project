#!/usr/bin/env python3

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

import torch
import torch.nn as nn

class Dueling_Neural_Network(nn.Module):
    """
    Dueling architecture:
    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """
    def __init__(self, state_dimension, action_dimension, hidden=128):
        super(Dueling_Neural_Network, self).__init__()

        # shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # value stream outputs scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # advantage stream outputs A(s,a) for each action
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dimension),
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)          # shape: (B, 1)
        advantages = self.adv_stream(features)        # shape: (B, A)

        # subtract mean advantage for identifiability
        advantages = advantages - advantages.mean(dim=1, keepdim=True)

        q_values = values + advantages                # broadcast (B,1) + (B,A)
        return q_values


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
    
    def __init__(self, env, learning_rate=3e-4, buffer_size=50000, batch_size=64, gamma=0.99, tensorboard_log=None):
        # Initialise parameters env, lr, batch_size, gamma and tensorboard_logs for metrics
        # Buffer Size of 50000 allows for more episodes to be stored, helps with learning unexpected events
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size= buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log

        # Hyperparameters, using epsilon decay to improve learning over episodes
        # Initially agent acts completely random, as training increases follows optimal policy as experience increases
        self.epsilon = 1.0
        self.end_epsilon = 0.05
        self.decay_rate = 0.005
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
        if random.random() < self.epsilon:  # Randomly select action if prob < epsilon
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

                # Epsilon Decay formula
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
            # Select best actions from policy network, calculate target_q values with these action from the target network
            best_next_actions = self.policy_network(next_states).argmax(1).unsqueeze(1)
            target_q = rewards + (self.gamma * self.target_network(next_states).gather(1, best_next_actions) * (1 - dones))

        huber_loss = nn.HuberLoss(delta=1.0)(policy_q, target_q)  # We normalise rewards with an upper bound of 1 hence delta of 1.0
        
        # Weights only updated for Policy Network using gradient descent, not for Target Network
        self.optim.zero_grad()
        huber_loss.backward()
        self.optim.step()
        
        return huber_loss.item()


class Dueling_DQN_Implementation:
    
    def __init__(self, env, learning_rate=3e-4, buffer_size=50000, batch_size=64,
                 gamma=0.99, tensorboard_log=None, hidden=128):
        
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log

       
        self.epsilon = 1.0
        self.end_epsilon = 0.05
        self.decay_rate = 0.005
        self.update_frequency = 1000
        self.steps_completed = 0

        
        self.experience_memory = Replay_Buffer(buffer_size)

        self.state_dimension = env.observation_space.shape[0]
        self.action_dimension = env.action_space.n

        
        self.policy_network = Dueling_Neural_Network(self.state_dimension, self.action_dimension, hidden=hidden)
        self.target_network = Dueling_Neural_Network(self.state_dimension, self.action_dimension, hidden=hidden)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optim = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        
        self.writer = None
        if self.tensorboard_log:
            os.makedirs(self.tensorboard_log, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

    def action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dimension)
        else:
            with torch.no_grad():
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

        
        policy_q = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            # vanilla DQN target (max over target net)
            next_q_max = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q_max * (1 - dones))

        huber_loss = nn.HuberLoss(delta=1.0)(policy_q, target_q)

        self.optim.zero_grad()
        huber_loss.backward()
        self.optim.step()

        return huber_loss.item()

    def learn(self, total_timesteps):
        print(f"Train for {total_timesteps} steps")
        state, _ = self.env.reset()
        episode_reward = 0

        for step in range(1, total_timesteps + 1):
            self.steps_completed += 1

            act = self.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(act)

            episode_reward += reward

            done_flag = terminated
            self.experience_memory.store_experience(state, act, reward, next_state, done_flag)

            state = next_state

            loss_val = self.update_network()

            if terminated or truncated:
                if self.writer:
                    self.writer.add_scalar("rollout/episode_reward_mean", episode_reward, self.steps_completed)
                    self.writer.add_scalar("train/epsilon", self.epsilon, self.steps_completed)

                print(f"Step: {self.steps_completed} | Episode Reward: {episode_reward:.3f} | Epsilon: {self.epsilon:.3f}")

                state, _ = self.env.reset()
                episode_reward = 0

                self.epsilon = max(self.end_epsilon, self.epsilon * (1 - self.decay_rate))

            if loss_val is not None and step % 100 == 0 and self.writer:
                self.writer.add_scalar("train/loss", loss_val, self.steps_completed)

            if self.steps_completed % self.update_frequency == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        print("Training finished")
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def save(self, path):
        if not path.endswith(".zip"):
            path += ".zip"

        print(f"Saving model to {path}")

        buffer = io.BytesIO()
        torch.save({
            'model_state_dictionary': self.policy_network.state_dict(),
            'optimizer_state_dictionary': self.optim.state_dict(),
            'epsilon': self.epsilon
        }, buffer)
        buffer.seek(0)

        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("policy.pth", buffer.read())

        print("Model saved")

########################################################################################################################################
# Distributional DQN

class Distributional_Neural_Network(nn.Module):
    def __init__(self, state_dimension, action_dimension, n_atoms=51, hidden=128):
        super().__init__()
        self.action_dimension = action_dimension
        self.n_atoms = n_atoms

        self.net = nn.Sequential(
            nn.Linear(state_dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dimension * n_atoms)
        )

    def forward(self, x):
        logits = self.net(x)  # (B, A*N)
        logits = logits.view(-1, self.action_dimension, self.n_atoms)  # (B, A, N)
        return logits

class Distributional_DQN_Implementation:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=None,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        hidden=128,
        update_frequency=1000,
        decay_rate=0.005,
        end_epsilon=0.05,
        device=None
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log

        # Epsilon-greedy params
        self.epsilon = 1.0
        self.end_epsilon = end_epsilon
        self.decay_rate = decay_rate

        self.update_frequency = update_frequency
        self.steps_completed = 0

        self.experience_memory = Replay_Buffer(buffer_size)

        self.state_dimension = env.observation_space.shape[0]
        self.action_dimension = env.action_space.n

        # Distribution support (atoms)
        self.n_atoms = n_atoms
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        z = torch.linspace(self.v_min, self.v_max, self.n_atoms)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.z = z.to(self.device)  # (N,)

        self.policy_network = Distributional_Neural_Network(
            self.state_dimension, self.action_dimension, n_atoms=self.n_atoms, hidden=hidden
        ).to(self.device)

        self.target_network = Distributional_Neural_Network(
            self.state_dimension, self.action_dimension, n_atoms=self.n_atoms, hidden=hidden
        ).to(self.device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optim = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # TensorBoard
        self.writer = None
        if self.tensorboard_log:
            os.makedirs(self.tensorboard_log, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

    def _dist(self, logits):
        # Convert logits -> probability distribution over atoms
        return torch.softmax(logits, dim=-1)  # (..., N)

    def _expected_q(self, logits):
        # E[Z] for each action: sum p(z_i) * z_i
        probs = self._dist(logits)  # (B, A, N)
        q = torch.sum(probs * self.z.view(1, 1, -1), dim=-1)  # (B, A)
        return q

    def action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dimension)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                logits = self.policy_network(s)  # (1, A, N)
                q_vals = self._expected_q(logits)  # (1, A)
                return q_vals.argmax(dim=1).item()

    @torch.no_grad()
    def _project_distribution(self, rewards, dones, next_logits):
        """
        C51 projection:
        Tz = r + gamma*(1-done)*z
        Project onto fixed support [v_min, v_max]
        """
        # next_logits: (B, A, N)
        next_probs = self._dist(next_logits)  # (B, A, N)

        # Greedy action by expected value under target distributions
        next_q = torch.sum(next_probs * self.z.view(1, 1, -1), dim=-1)  # (B, A)
        next_actions = next_q.argmax(dim=1, keepdim=True)  # (B, 1)

        # Pick distribution for best action: (B, N)
        next_probs_a = next_probs.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)).squeeze(1)

        # Bellman update on atoms
        # rewards, dones: (B, 1)
        Tz = rewards + self.gamma * (1.0 - dones) * self.z.view(1, -1)  # (B, N)
        Tz = torch.clamp(Tz, self.v_min, self.v_max)

        # Compute projection locations
        b = (Tz - self.v_min) / self.delta_z  # (B, N)
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        # Fix possible numerical issues where l==u
        l = torch.clamp(l, 0, self.n_atoms - 1)
        u = torch.clamp(u, 0, self.n_atoms - 1)

        m = torch.zeros_like(next_probs_a)  # (B, N)

        # Distribute probability mass
        # m[l] += p * (u - b)
        # m[u] += p * (b - l)
        offset = torch.arange(next_probs_a.size(0), device=self.device).unsqueeze(1)  # (B,1)

        m.view(-1).index_add_(
            0,
            (l + offset * self.n_atoms).view(-1),
            (next_probs_a * (u.float() - b)).view(-1)
        )
        m.view(-1).index_add_(
            0,
            (u + offset * self.n_atoms).view(-1),
            (next_probs_a * (b - l.float())).view(-1)
        )

        # m is the projected target distribution (B, N)
        return m

    def update_network(self):
        if len(self.experience_memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.experience_memory.mini_batch(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)   # (B,1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (B,1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)      # (B,1)

        # Current logits for all actions
        logits = self.policy_network(states)  # (B, A, N)
        # Select logits for taken actions: (B, N)
        logits_a = logits.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)).squeeze(1)

        # Target projected distribution m: (B, N)
        with torch.no_grad():
            next_logits = self.target_network(next_states)  # (B, A, N)
            target_dist = self._project_distribution(rewards, dones, next_logits)  # (B, N)

        # Cross-entropy loss: - sum m * log p
        log_probs = torch.log_softmax(logits_a, dim=-1)  # (B, N)
        loss = -(target_dist * log_probs).sum(dim=-1).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()
    def save(self, path):
        if not path.endswith(".zip"):
            path += ".zip"

        print(f"Saving model to {path}")

        buffer = io.BytesIO()
        torch.save({
            'model_state_dictionary': self.policy_network.state_dict(),
            'optimizer_state_dictionary': self.optim.state_dict(),
            'epsilon': self.epsilon
        }, buffer)
        buffer.seek(0)

        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("policy.pth", buffer.read())

        print("Model saved")


    def learn(self, total_timesteps):
        print(f"Train for {total_timesteps} steps | device={self.device}")
        state, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(1, total_timesteps + 1):
            self.steps_completed += 1

            act = self.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(act)

            episode_reward += reward

            done_flag = terminated  # follow your convention
            self.experience_memory.store_experience(state, act, reward, next_state, done_flag)
            state = next_state

            loss_val = self.update_network()

            if terminated or truncated:
                if self.writer:
                    self.writer.add_scalar("rollout/episode_reward_mean", episode_reward, self.steps_completed)
                    self.writer.add_scalar("train/epsilon", self.epsilon, self.steps_completed)

                print(f"Step: {self.steps_completed} | Episode Reward: {episode_reward:.3f} | Epsilon: {self.epsilon:.3f}")

                state, _ = self.env.reset()
                episode_reward = 0.0

                self.epsilon = max(self.end_epsilon, self.epsilon * (1 - self.decay_rate))

            if loss_val is not None and step % 100 == 0 and self.writer:
                self.writer.add_scalar("train/loss", loss_val, self.steps_completed)

            if self.steps_completed % self.update_frequency == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        print("Training finished")
        if self.writer:
            self.writer.flush()
            self.writer.close()

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
    env = NormaliseObservation(env)

    agent = DuellingDQN_Implementation(env, tensorboard_log="../logs/DuellingDQN_SMA_Normalised")
    agent.learn(total_timesteps=200000)
    agent.save("../models/DuellingDQN_SMA_Normalised.zip")
    print(f"\nTraining complete. Model saved to ../models/DuellingDQN_SMA_Normalised.zip")
    print(f"View logs: tensorboard --logdir=../logs")
