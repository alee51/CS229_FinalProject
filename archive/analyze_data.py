import numpy as np
import os

os.chdir('attempt 1')
data = np.load('expert_data_reach-v3.npz', allow_pickle=True)
states = data['states']
actions = data['actions']

print(f'Num trajectories: {len(states)}')
print(f'State shapes (first 5): {[s.shape for s in states[:5]]}')
print(f'Action shapes (first 5): {[a.shape for a in actions[:5]]}')
print(f'Avg trajectory length: {np.mean([len(s) for s in states]):.1f}')
print(f'Min/Max trajectory length: {min(len(s) for s in states)} / {max(len(s) for s in states)}')
print(f'State dim: {states[0].shape[1] if len(states[0].shape) > 1 else "1D"}')
print(f'Action dim: {actions[0].shape[1] if len(actions[0].shape) > 1 else "1D"}')
print(f'Total samples (all trajectories concatenated): {sum(len(s) for s in states)}')

# Check state/action value ranges
all_states = np.concatenate(states)
all_actions = np.concatenate(actions)
print(f'\nState ranges: min={all_states.min():.3f}, max={all_states.max():.3f}')
print(f'Action ranges: min={all_actions.min():.3f}, max={all_actions.max():.3f}')
