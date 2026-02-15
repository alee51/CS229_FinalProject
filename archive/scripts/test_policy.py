import metaworld
import torch
import numpy as np
from train import ClonePolicy

def test_student_robot(task_name='reach-v3'):
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]() # Headless for speed
    
    model = ClonePolicy(39, 4)
    model.load_state_dict(torch.load('cloned_policy_stable2.pth'))
    model.eval()

    success_count = 0
    num_episodes = 1000

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

    print(f"📊 Results: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")

if __name__ == "__main__":
    test_student_robot()