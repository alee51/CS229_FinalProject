import metaworld
import torch
import numpy as np
import sys
sys.path.append('.')
from train import ClonePolicy

mt1 = metaworld.MT1('reach-v3')
env = mt1.train_classes['reach-v3']()

model = ClonePolicy(39, 4)
model.load_state_dict(torch.load('cloned_policy_stable2.pth'))
model.eval()

success_count = 0
num_episodes = 100

for i in range(num_episodes):
    task = mt1.train_tasks[i % len(mt1.train_tasks)]
    env.set_task(task)
    obs, info = env.reset()
    done = False
    
    while not done:
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            action = model(obs_tensor).numpy()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if info['success']:
        success_count += 1

success_rate = success_count/num_episodes*100
print(f"cloned_policy_stable2.pth (actual existing model): {success_rate:.1f}%")
