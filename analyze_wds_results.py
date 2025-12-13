import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Episode stats
stats = pd.read_csv("evaluations/episode_stats.csv")

# (2) Performance bar chart (one agent = one bar)
sns.barplot(data=stats, y="total_reward")
plt.title("PPO WDSEnv: Episode Returns")
plt.show()

# (3) Episode trajectory (water level)
traj = pd.read_csv("evaluations/episode0_trajectory.csv")
fig = px.line(traj, x="step", y="tank_level_0", title="Water Level – Episode 0")
fig.show()

# (4) Statistical table
print(stats["total_reward"].describe())

# (5) Distribution plot
sns.violinplot(data=stats, y="total_reward")
plt.title("Reward Distribution")
plt.show()

# (6) Heatmap: reshape rewards as 2D grid if desired
# For example, 10x5 matrix (episodes x dummy dimension)
import numpy as np
mat = np.array(stats["total_reward"][:50]).reshape(10, 5)
sns.heatmap(mat, annot=False)
plt.title("Reward Heatmap (episodes grouped)")
plt.show()
