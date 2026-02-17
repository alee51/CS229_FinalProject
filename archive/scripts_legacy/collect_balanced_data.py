"""
Improved expert data collection that ensures coverage of all 50 reach-v3 goal variations
"""

import metaworld
import numpy as np
import os
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

def collect_expert_data_balanced(task_name='reach-v3', episodes_per_task=40, output_dir='baseline/data'):
    """
    Collect expert data by iterating through all train_tasks to ensure balanced coverage
    
    With 40 episodes per task × 50 tasks = 2000 total episodes
    This ensures every goal variation is well represented
    """
    
    print("="*70)
    print("COLLECTING BALANCED EXPERT DATA FOR REACH-V3")
    print("="*70)
    
    # Setup
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    policy = SawyerReachV3Policy()
    
    print(f"\nConfiguration:")
    print(f"  Number of goal variations: {len(mt1.train_tasks)}")
    print(f"  Episodes per task: {episodes_per_task}")
    print(f"  Total episodes target: {len(mt1.train_tasks) * episodes_per_task}")
    print()
    
    all_states = []
    all_actions = []
    total_success = 0
    total_failures = 0
    task_results = {}
    
    # Iterate through all 50 train_tasks explicitly
    for task_idx, task in enumerate(mt1.train_tasks):
        task_success = 0
        task_attempts = 0
        
        for ep in range(episodes_per_task):
            env.set_task(task)
            obs, info = env.reset()
            done = False
            steps = 0
            
            episode_states = []
            episode_actions = []
            
            while not done and steps < 500:
                action = policy.get_action(obs)
                episode_states.append(obs)
                episode_actions.append(action)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                
                if info['success']:
                    break
            
            task_attempts += 1
            
            if info['success']:
                all_states.append(np.array(episode_states))
                all_actions.append(np.array(episode_actions))
                task_success += 1
                total_success += 1
            else:
                total_failures += 1
        
        task_results[task_idx] = (task_success, task_attempts)
        
        # Progress
        pct_done = (task_idx + 1) / len(mt1.train_tasks) * 100
        print(f"  Task {task_idx:2d}/{len(mt1.train_tasks)-1}: {task_success:2d}/{task_attempts} successful | "
              f"Total: {total_success:4d} episodes | {pct_done:5.1f}% complete")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'expert_data_{task_name}_balanced.npz')
    
    print(f"\n{'='*70}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total successful episodes: {total_success}")
    print(f"Total failed attempts: {total_failures}")
    print(f"Success rate: {total_success / (total_success + total_failures) * 100:.1f}%")
    
    # Task-level stats
    successes = [s for s, _ in task_results.values()]
    print(f"\nPer-task success rate:")
    print(f"  Min: {min(successes)}/{episodes_per_task}")
    print(f"  Max: {max(successes)}/{episodes_per_task}")
    print(f"  Mean: {np.mean(successes):.1f}/{episodes_per_task}")
    
    # Identify problematic tasks
    problem_tasks = [idx for idx, (s, _) in task_results.items() if s < episodes_per_task * 0.8]
    if problem_tasks:
        print(f"\nTasks with <80% success rate:")
        for idx in problem_tasks[:5]:
            s, a = task_results[idx]
            print(f"  Task {idx}: {s}/{a}")
        if len(problem_tasks) > 5:
            print(f"  ... and {len(problem_tasks) - 5} more")
    
    print(f"\nSaving to: {filename}")
    np.savez(filename, 
             states=np.array(all_states, dtype=object), 
             actions=np.array(all_actions, dtype=object))
    print(f"✅ Done! Expert data saved.\n")
    
    return filename


def replace_original_data(new_data_path, backup=True):
    """Replace the original expert data with the new balanced version"""
    
    original_path = 'baseline/data/expert_data_reach-v3.npz'
    
    if backup and os.path.exists(original_path):
        backup_path = original_path.replace('.npz', '_backup.npz')
        print(f"Backing up original to: {backup_path}")
        os.rename(original_path, backup_path)
    
    print(f"Replacing with: {new_data_path}")
    os.rename(new_data_path, original_path)
    print(f"✅ Expert data replaced!\n")


if __name__ == "__main__":
    # Collect balanced expert data
    new_data_path = collect_expert_data_balanced(
        task_name='reach-v3',
        episodes_per_task=40,
        output_dir='baseline/data'
    )
    
    # Ask user before replacing
    response = input("Replace original expert_data_reach-v3.npz? (y/n): ").strip().lower()
    if response == 'y':
        replace_original_data(new_data_path, backup=True)
        print("Ready to train on new balanced expert data!")
    else:
        print(f"New data saved as: {new_data_path}")
        print("Original data unchanged.")
