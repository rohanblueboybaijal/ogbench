import ogbench
import numpy as np
import pickle
import os

# Make an environment and datasets (they will be automatically downloaded).
dataset_name = 'cube-single-play-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

# Train your offline goal-conditioned RL agent on the dataset.
# ...
print(train_dataset.keys())
print(train_dataset['observations'].shape)
print(train_dataset['actions'].shape)

ccil_dataset = []
trajectory = dict()
trajectory['observations'] = []
trajectory['actions'] = []
num_obs = 0

for i in range(train_dataset['observations'].shape[0]):
    trajectory['observations'].append(train_dataset['observations'][i])
    trajectory['actions'].append(train_dataset['actions'][i])
    num_obs += 1
    if train_dataset['terminals'][i] == 1:
        trajectory['observations'] = np.array(trajectory['observations'])
        trajectory['actions'] = np.array(trajectory['actions'])
        ccil_dataset.append(trajectory)
        trajectory = dict()
        trajectory['observations'] = []
        trajectory['actions'] = []
    
print(f"Number of observations: {num_obs}")
print(f"No. of trajectories: {len(ccil_dataset)}")

dataset_dir = "ccil_datasets"
save_path = os.path.join(dataset_dir, f"{dataset_name}.pkl")

print(f"Saving dataset to {save_path}")
os.makedirs(dataset_dir, exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(ccil_dataset, f)