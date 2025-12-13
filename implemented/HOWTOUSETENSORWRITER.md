# How to use Tensor Writer

1. Import 
```python
from torch.utils.tensorboard import SummaryWriter 
```
2. Init
``` python
writer = SummaryWriter(f"{log_dir}/{run_name}")
```
`log_dir/run_name` SHOULD be passed during model Init

Example:
```python
390     agent = train_ppo(env, total_timesteps=200000, log_dir="../logs")
391     agent.save("../models/ppo_scratch.pt")
```

3. Log stuff
```python
# Scalars (most common)
writer.add_scalar('train/loss', loss_value, step)
writer.add_scalar('rollout/reward', reward, step)

# Multiple scalars on same plot
writer.add_scalars('comparison', {'train': train_loss, 'val': val_loss}, step)

# Text (hyperparams, notes)
writer.add_text('config', 'lr=3e-4, gamma=0.99')

# Histogram (weight distributions)
writer.add_histogram('policy/weights', model.fc.weight, step)

# Close when done
writer.close()
```

4. View
```bash
tensorboard --logdir=logs/
```
Should start a server at localhost:6006
