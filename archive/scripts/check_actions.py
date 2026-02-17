import numpy as np

data = np.load('expert_data_reach-v3.npz', allow_pickle=True)
actions = data['actions']
all_actions = np.concatenate(actions)

print('Action statistics:')
print(f'  Min per dim: {all_actions.min(axis=0)}')
print(f'  Max per dim: {all_actions.max(axis=0)}')
print(f'  % within [-1,1]: {(np.abs(all_actions) <= 1.0).mean() * 100:.1f}%')
print(f'  % with any dim outside [-1,1]: {(np.abs(all_actions).max(axis=1) > 1.0).mean() * 100:.1f}%')
