import metaworld
import torch
import numpy as np
import sys
sys.path.append('.')
from train_improved import ClonePolicy

def test_student_robot_improved(task_name='reach-v3'):
    """Test with proper action denormalization"""
    
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    
    # Load model and normalization parameters
    checkpoint = torch.load('baseline_improved.pth')
    model = ClonePolicy(39, 4)
    model.load_state_dict(checkpoint['policy_state'])
    action_min = checkpoint['action_min']
    action_max = checkpoint['action_max']
    model.eval()
    
    print(f"Evaluating on {task_name}...")
    print(f"Action normalization params: min={action_min.flatten()}, max={action_max.flatten()}")

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
                action_normalized = model(obs_tensor).numpy()
            
            # CRITICAL FIX: Denormalize actions from [-1, 1] to original range
            action = (action_normalized + 1) / 2 * (action_max - action_min) + action_min
            action = np.clip(action, -1.0, 1.0)  # Clip to valid action space
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info['success']:
            success_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/1000 | Current rate: {success_count/(i+1)*100:.1f}%")

    final_rate = success_count/num_episodes*100
    print(f"\n📊 Final Results: {success_count}/{num_episodes} ({final_rate:.1f}%)")
    return final_rate

if __name__ == "__main__":
    test_student_robot_improved()
