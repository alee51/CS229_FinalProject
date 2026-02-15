import metaworld
import torch
import numpy as np
import argparse
import sys
import os

# Add parent directory to path so we can import train module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import ClonePolicy

def test_policy(model_path, num_episodes=100, task_name='reach-v3', clip_actions=False, verbose=False):
    """Test a policy model
    
    Args:
        model_path: Path to the saved policy model (can be relative or absolute)
        num_episodes: Number of episodes to test
        task_name: MetaWorld task name
        clip_actions: Whether to clip actions to [-1, 1]
        verbose: Print progress every N episodes
    
    Returns:
        success_rate: Success rate as a percentage
    """
    
    # Resolve model path - if relative, look in ../models/
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '..', 'models', model_path)
    
    try:
        mt1 = metaworld.MT1(task_name)
        env = mt1.train_classes[task_name]()
    except Exception as e:
        print(f"❌ Failed to load task '{task_name}': {e}")
        return None
    
    try:
        model = ClonePolicy(39, 4)
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"❌ Failed to load model '{model_path}': {e}")
        return None
    
    model.eval()
    success_count = 0
    
    for i in range(num_episodes):
        task = mt1.train_tasks[i % len(mt1.train_tasks)]
        env.set_task(task)
        obs, info = env.reset()
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action = model(obs_tensor).numpy()
            
            if clip_actions:
                action = np.clip(action, -1.0, 1.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info['success']:
            success_count += 1
        
        if verbose and (i + 1) % verbose == 0:
            print(f"  Progress: {i+1}/{num_episodes} | Current rate: {success_count/(i+1)*100:.1f}%")

    success_rate = success_count/num_episodes*100
    return success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a behavioral cloning policy')
    parser.add_argument('--model', type=str, required=True, help='Path or name of the policy model (.pth file)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of test episodes (default: 100)')
    parser.add_argument('--task', type=str, default='reach-v3', help='MetaWorld task name (default: reach-v3)')
    parser.add_argument('--clip', action='store_true', help='Clip actions to [-1, 1]')
    parser.add_argument('--verbose', type=int, default=0, help='Print progress every N episodes (0=off)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Testing Policy")
    print(f"{'='*60}")
    print(f"Model:           {args.model}")
    print(f"Task:            {args.task}")
    print(f"Episodes:        {args.episodes}")
    print(f"Clip Actions:    {args.clip}")
    print(f"{'='*60}\n")
    
    result = test_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        task_name=args.task,
        clip_actions=args.clip,
        verbose=args.verbose
    )
    
    if result is not None:
        print(f"\n✅ Success Rate: {result:.2f}%")
    else:
        print(f"\n❌ Test failed")
