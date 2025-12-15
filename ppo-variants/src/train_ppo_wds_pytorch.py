"""
Custom PPO (PyTorch) training script for the WaterDistributionSystemEnv (WDSEnv)
environment from the Gym4ReaL benchmark suite.

Environment reference:
  Davide Salaorni et al., "Gym4ReaL: A Suite for Benchmarking Real-World
  Reinforcement Learning", arXiv:2507.00257, 2025.
  WDSEnv models a municipal water distribution system controlled via EPANET,
  as described in Section 2.6 and Appendix D.6 of the paper.

This script implements a from-scratch PPO agent (Actor-Critic, GAE, clipped
objective) to control WDSEnv using the Gymnasium-compatible interface
`gym.make("gym4real/wds-v0", settings=...)`.
"""


import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import argparse
from datetime import datetime
import json
import zipfile
import tempfile
import shutil


# ==================== ACTOR-CRITIC NETWORK ====================
class ActorCritic(nn.Module):
    """Combined Actor-Critic network for PPO."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[128, 128]):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh()
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dims[1], action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dims[1], 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        """Forward pass through both actor and critic."""
        features = self.shared_net(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action_and_value(self, obs, action=None):
        """Get action, log probability, entropy, and value."""
        action_logits, value = self.forward(obs)
        probs = Categorical(logits=action_logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value


# ==================== PPO AGENT ====================
class PPOAgent:
    """PPO Agent with training logic."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64
    ):
        self.device = device
        self.model = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Training statistics
        self.total_steps = 0

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def collect_rollout(self, env, n_steps):
        """Collect rollout data from environment."""
        observations, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        obs, _ = env.reset()

        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated

            observations.append(obs)
            actions.append(action.cpu().item())
            log_probs.append(log_prob.cpu().item())
            values.append(value.cpu().item())
            rewards.append(reward)
            dones.append(done)

            obs = next_obs
            self.total_steps += 1

            if done:
                obs, _ = env.reset()

        # Get next value for GAE
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, next_value = self.model.get_action_and_value(obs_tensor)
        next_value = next_value.cpu().item()

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        return {
            'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'returns': np.array(returns, dtype=np.float32),
            'advantages': np.array(advantages, dtype=np.float32),
            'rewards': rewards
        }

    def update(self, rollout_data):
        """Update policy using PPO loss."""
        obs = torch.FloatTensor(rollout_data['observations']).to(self.device)
        actions = torch.LongTensor(rollout_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout_data['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout_data['advantages']).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        policy_losses, value_losses, entropy_losses = [], [], []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                _, log_prob, entropy, value = self.model.get_action_and_value(
                    obs[batch_indices],
                    actions[batch_indices]
                )

                # PPO clipped objective
                ratio = torch.exp(log_prob - old_log_probs[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value.squeeze(), returns[batch_indices])

                # Entropy bonus (maximize entropy => minimize negative entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            'policy_loss': float(np.mean(policy_losses)),
            'value_loss': float(np.mean(value_losses)),
            'entropy_loss': float(np.mean(entropy_losses)),
            'mean_reward': float(np.mean(rollout_data['rewards']))
        }

    def save_pth(self, path):
        """Save model checkpoint in plain PyTorch format."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)

    def load_pth(self, path, map_location=None):
        """Load model checkpoint from plain PyTorch format."""
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']


# ==================== SB3-LIKE ZIP SAVE ====================
def save_sb3_like_zip(agent: PPOAgent, config: dict, path_zip: str):
    """
    Save agent in a Stable-Baselines3-like .zip archive:
    policy.pth, policy.optimizer.pth, pytorch_variables.pth, data.json.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1) save weights / optimizer / extra vars
        torch.save(agent.model.state_dict(), os.path.join(tmp_dir, "policy.pth"))
        torch.save(agent.optimizer.state_dict(), os.path.join(tmp_dir, "policy.optimizer.pth"))
        torch.save({'total_steps': agent.total_steps}, os.path.join(tmp_dir, "pytorch_variables.pth"))

        # 2) save config as JSON (similar to SB3 data dict)
        with open(os.path.join(tmp_dir, "data.json"), "w") as f:
            json.dump(config, f, indent=4)

        # 3) zip everything
        with zipfile.ZipFile(path_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmp_dir):
                full_path = os.path.join(tmp_dir, fname)
                zf.write(full_path, arcname=fname)
    finally:
        shutil.rmtree(tmp_dir)


# ==================== ENVIRONMENT SETUP ====================
def create_wds_env():
    """Create WDS environment with correct absolute paths (Windows safe)."""

    from gym4real.envs.wds.utils import parameter_generator

    # Absolute path to local gym4ReaL package
    base_path = os.path.join(os.path.dirname(__file__), "gym4ReaL")

    # Change cwd so YAML relative imports work
    original_cwd = os.getcwd()
    try:
        os.chdir(base_path)
        params = parameter_generator(
            world_options="gym4real/envs/wds/world_anytown.yaml"
        )

        # Fix absolute paths
        inp_file = params.get("inp_file")
        if isinstance(inp_file, str) and not os.path.isabs(inp_file):
            params["inp_file"] = os.path.join(base_path, inp_file)

        # Fix demand dataset path
        if 'demand' in params and 'data_config' in params['demand']:
            dataset_path = params['demand']['data_config'].get('dataset_path', '')
            if dataset_path and not os.path.isabs(dataset_path):
                params['demand']['data_config']['dataset_path'] = os.path.join(base_path, dataset_path)

    finally:
        os.chdir(original_cwd)

    # Create Gym environment
    env = gym.make("gym4real/wds-v0", settings=params)

    return env


# ==================== TRAINING LOOP ====================
def train_wds_ppo_pytorch(
    total_timesteps=100000,
    n_steps=2048,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    save_path="models/ppo_wds_pytorch"
):
    """
    Train a PPO model (PyTorch) on the WDS environment.
    """

    print("=" * 70)
    print("Training Custom PyTorch PPO on WDS Environment")
    print("=" * 70)

    # Import gym4real
    try:
        import gym4real  # noqa: F401
    except ImportError:
        print("\nWarning: gym4real package not found. Adding to path...")
        gym4real_path = os.path.join(os.path.dirname(__file__), 'gym4ReaL')
        if os.path.exists(gym4real_path):
            sys.path.insert(0, gym4real_path)
            import gym4real  # noqa: F401
            print("Successfully imported gym4real.")
        else:
            raise ImportError("Could not find gym4real package.")

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    try:
        # Create environment
        print("\nCreating WDS environment...")
        env = create_wds_env()

        print(f"Environment created")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Initialize agent
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_range,
            entropy_coef=ent_coef,
            n_epochs=n_epochs,
            batch_size=batch_size
        )

        print(f"\nPPO Agent initialized")
        print(f"  Model parameters: {sum(p.numel() for p in agent.model.parameters())}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Steps per update: {n_steps}")

        # TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f"logs/tensorboard/ppo_pytorch_{timestamp}")

        # Training loop
        print(f"\nStarting training for {total_timesteps} timesteps...")
        print("=" * 70)

        n_updates = total_timesteps // n_steps
        best_mean_reward = -float('inf')

        for update in range(n_updates):
            # Collect rollout
            rollout_data = agent.collect_rollout(env, n_steps)

            # Update policy
            losses = agent.update(rollout_data)

            # Log metrics vs update index
            writer.add_scalar('train/policy_loss', losses['policy_loss'], update)
            writer.add_scalar('train/value_loss', losses['value_loss'], update)
            writer.add_scalar('train/entropy_loss', losses['entropy_loss'], update)
            writer.add_scalar('train/mean_reward', losses['mean_reward'], update)

            # Also log vs total timesteps (more natural x-axis)
            writer.add_scalar('rollout/mean_step_reward',
                              float(np.mean(rollout_data['rewards'])),
                              agent.total_steps)
            writer.add_scalar('rollout/std_step_reward',
                              float(np.std(rollout_data['rewards'])),
                              agent.total_steps)
            writer.add_scalar('rollout/steps_in_rollout',
                              len(rollout_data['rewards']),
                              agent.total_steps)
            current_lr = agent.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', current_lr, agent.total_steps)

            # Print progress
            if (update + 1) % 10 == 0 or update == 0:
                print(f"\nUpdate {update + 1}/{n_updates}")
                print(f"  Total Steps: {agent.total_steps}")
                print(f"  Mean Reward (rollout): {losses['mean_reward']:.2f}")
                print(f"  Policy Loss: {losses['policy_loss']:.4f}")
                print(f"  Value Loss: {losses['value_loss']:.4f}")
                print(f"  Entropy Loss: {losses['entropy_loss']:.4f}")
                print("-" * 70)

            # Save best model (pth + zip)
            if losses['mean_reward'] > best_mean_reward:
                best_mean_reward = losses['mean_reward']
                best_model_pth = f"{save_path}_best.pth"
                agent.save_pth(best_model_pth)
                if verbose:
                    print(f"  New best model saved! Mean reward: {best_mean_reward:.2f}")

                # Build config dict for saving
                best_config = {
                    'total_timesteps': int(total_timesteps),
                    'n_steps': int(n_steps),
                    'learning_rate': float(learning_rate),
                    'batch_size': int(batch_size),
                    'n_epochs': int(n_epochs),
                    'gamma': float(gamma),
                    'gae_lambda': float(gae_lambda),
                    'clip_range': float(clip_range),
                    'ent_coef': float(ent_coef),
                    'obs_dim': int(obs_dim),
                    'action_dim': int(action_dim),
                    'best_mean_reward': float(best_mean_reward)
                }
                best_zip_path = f"{save_path}_best.zip"
                save_sb3_like_zip(agent, best_config, best_zip_path)

            # Save checkpoint every 10 updates (.pth only to avoid many zips)
            if (update + 1) % 10 == 0:
                checkpoint_path = f"{save_path}_checkpoint_{update+1}.pth"
                agent.save_pth(checkpoint_path)

        # Final model save (.pth + config JSON + zip)
        print(f"\nSaving final model to {save_path}.pth...")
        agent.save_pth(f"{save_path}.pth")

        config = {
            'total_timesteps': int(total_timesteps),
            'n_steps': int(n_steps),
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'n_epochs': int(n_epochs),
            'gamma': float(gamma),
            'gae_lambda': float(gae_lambda),
            'clip_range': float(clip_range),
            'ent_coef': float(ent_coef),
            'obs_dim': int(obs_dim),
            'action_dim': int(action_dim),
            'best_mean_reward': float(best_mean_reward)
        }

        # Save config JSON
        with open(f"{save_path}_config.json", 'w') as f:
            json.dump(config, f, indent=4)

        # Save SB3-like zip
        zip_path = f"{save_path}.zip"
        save_sb3_like_zip(agent, config, zip_path)

        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"  Total steps trained: {agent.total_steps}")
        print(f"  Best mean reward: {best_mean_reward:.2f}")
        print(f"  Model (.pth) saved to: {save_path}.pth")
        print(f"  Best model (.pth) saved to: {save_path}_best.pth")
        print(f"  Final SB3-like zip saved to: {zip_path}")
        print(f"  Best SB3-like zip saved to: {save_path}_best.zip")
        print(f"  Config saved to: {save_path}_config.json")
        print(f"  TensorBoard logs: logs/tensorboard/")
        print("\nTo view training curves, run:")
        print("  tensorboard --logdir logs/tensorboard/")
        print("=" * 70)

        writer.close()
        env.close()

        return agent

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== EVALUATION ====================
def evaluate_ppo_pytorch(model_path, n_episodes=10):
    """Evaluate trained PyTorch PPO model (.pth path expected)."""

    print("=" * 70)
    print("Evaluating PyTorch PPO Model")
    print("=" * 70)

    import gym4real  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = model_path.replace('.pth', '_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create environment
    env = create_wds_env()

    # Initialize agent
    agent = PPOAgent(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        device=device
    )

    # Load model
    agent.load_pth(model_path, map_location=device)
    agent.model.eval()

    # Evaluate
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _, _ = agent.model.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    print("\n" + "=" * 70)
    print("Evaluation Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 70)

    env.close()

    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths))
    }


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO (PyTorch) on WDS environment")

    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per rollout"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for PPO updates"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a trained model instead of training"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_wds_pytorch.pth",
        help="Path to save/load model (.pth)"
    )

    args = parser.parse_args()

    if args.eval:
        # Evaluation mode
        evaluate_ppo_pytorch(args.model_path, n_episodes=10)
    else:
        # Training mode
        train_wds_ppo_pytorch(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            save_path="models/ppo_wds_pytorch"
        )
