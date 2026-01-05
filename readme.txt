REINFORCEMENT LEARNING FOR WATER DISTRIBUTION SYSTEM CONTROL
=============================================================

SETUP
-----

1. Clone and install the environment

    git clone https://github.com/Daveonwave/gym4ReaL/
    cd gym4ReaL/
    pip install -e . --break-system-packages
    cd ..

2. Install dependencies

    pip install -r requirements.txt --break-system-packages

3. Move reward wrapper to environment

    cp algorithms/reward_scaling_wrapper.py gym4ReaL/gym4real/envs/wds/


TRAINING
--------

Run algorithms from the project root:

    python3 algorithms/PPO.py
    python3 algorithms/TRPO.py
    python3 algorithms/DQN.py
    python3 algorithms/A3C.py


MONITORING TRAINING
-------------------

    tensorboard --logdir logs/


ENVIRONMENT DETAILS
-------------------

Observation space (27 dimensions):
- Tank water levels (2)
- Node pressures (22)
- Demand forecast values (2)
- Time encoding (1)

Action space (Discrete, 4 actions):
- Pump on/off combinations for the 2 pumps

Reward: Based on maintaining pressure bounds and minimizing tank overflow

CLONED ENVIRONMENT
------------------

gym4ReaL: https://github.com/Daveonwave/gym4ReaL
Gymnasium environment for water networks
