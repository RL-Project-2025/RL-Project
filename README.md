# Reinforcement Learning for Water Distribution System Control

## Setup

### 1. Clone and install the environment

```bash
git clone https://github.com/Daveonwave/gym4ReaL/
cd gym4ReaL/
pip install -e . --break-system-packages
cd ..
```

### 2. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Move reward wrapper to environment

```bash
cp algorithms/reward_scaling_wrapper.py gym4ReaL/gym4real/envs/wds/
```

## Training

Run algorithms from the project root:

```bash
# Example train 
python3 algorithms/PPO.py
python3 algorithms/TRPO.py
python3 algorithms/DQN.py
python3 algorithms/A2C.py
```

## Monitoring Training

```bash
tensorboard --logdir logs/
```

## Environment Details

**Observation space** (27 dimensions):

- Tank water levels (2)
- Node pressures (22)
- Demand forecast values (2)
- Time encoding (1)

**Action space** (Discrete, 4 actions):

- Pump on/off combinations for the 2 pumps

**Reward**: Based on maintaining pressure bounds and minimizing tank overflow

## Cloned environment
[gym4ReaL](https://github.com/Daveonwave/gym4ReaL) - Gymnasium environment for water networks
