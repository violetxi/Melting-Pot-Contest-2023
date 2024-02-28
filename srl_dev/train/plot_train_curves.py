import wandb
import matplotlib.pyplot as plt

project_name = 'pd_repeated_torch'
entity = 'stanford_autonomous_agent'
api = wandb.Api()

# List of runs you want to plot, replace these with your actual run IDs or names
runs_ids = ['084b0_00000', '084b0_00001', '084b0_00002']

# Initialize a dict to hold the data
data = {}

# Fetch the runs and their data
for run_id in runs_ids:
    run = api.run(f"{entity}/{project_name}/{run_id}")
    history = run.scan_history(keys=["episode_reward_mean"])
    rewards = [x['episode_reward_mean'] for x in history]
    data[run_id] = rewards

# Now plot
plt.figure(figsize=(10, 6))
for run_id, rewards in data.items():
    plt.plot(rewards, label=run_id)

plt.title('Training Reward Curves')
plt.xlabel('Epochs')
plt.ylabel('Reward')
plt.legend()
# save plot in results/figures folder
plt.savefig('/ccn2/u/locross/Melting-Pot-Contest-2023/srl_dev/results/figures/train_curves.png')
plt.show()
