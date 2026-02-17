"""
Expert data collection with proper train/test split
Train on 40 goal variations, test generalization on 10 held-out goals
"""

import metaworld
import numpy as np
import os
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

def collect_expert_data_with_split(task_name='reach-v3', 
                                    episodes_per_goal=40,
                                    train_goals=40, 
                                    test_goals=10,
                                    output_dir='baseline/data'):
    """
    Collect expert data with train/test split on GOAL VARIATIONS
    
    This ensures the model learns to REACH (general skill) not memorize 
    specific goal trajectories.
    
    Args:
        episodes_per_goal: How many expert demos per goal variation
        train_goals: How many goal variations to train on (40)
        test_goals: How many goal variations to hold out for testing (10)
    """
    
    assert train_goals + test_goals <= 50, "Total goals must be <= 50"
    
    print("="*70)
    print("COLLECTING EXPERT DATA WITH TRAIN/TEST GOAL SPLIT")
    print("="*70)
    
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    policy = SawyerReachV3Policy()
    
    # Shuffle and split goals
    all_goal_indices = np.arange(len(mt1.train_tasks))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(all_goal_indices)
    
    train_goal_indices = all_goal_indices[:train_goals]
    test_goal_indices = all_goal_indices[train_goals:train_goals + test_goals]
    
    print(f"\nConfiguration:")
    print(f"  Total goal variations available: {len(mt1.train_tasks)}")
    print(f"  Training goal variations: {train_goals}")
    print(f"  Test goal variations (held-out): {test_goals}")
    print(f"  Episodes per goal: {episodes_per_goal}")
    print(f"  Total training episodes: {train_goals * episodes_per_goal}")
    print()
    
    # Collect training data
    print(f"Collecting TRAINING data ({train_goals} goal variations)...")
    train_states = []
    train_actions = []
    
    for task_idx, goal_idx in enumerate(train_goal_indices):
        task = mt1.train_tasks[goal_idx]
        goal_success = 0
        
        for ep in range(episodes_per_goal):
            env.set_task(task)
            obs, _ = env.reset()
            done = False
            
            episode_states = []
            episode_actions = []
            
            while not done and len(episode_states) < 500:
                action = policy.get_action(obs)
                episode_states.append(obs)
                episode_actions.append(action)
                
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if info['success']:
                    break
            
            if info['success']:
                train_states.append(np.array(episode_states))
                train_actions.append(np.array(episode_actions))
                goal_success += 1
        
        if (task_idx + 1) % 10 == 0 or task_idx == 0:
            pct = (task_idx + 1) / train_goals * 100
            print(f"  Goal {task_idx + 1:2d}/{train_goals}: {goal_success}/{episodes_per_goal} successful | {pct:5.1f}%")
    
    # Collect test data (same distribution as training to verify)
    print(f"\nCollecting TEST data ({test_goals} goal variations)...")
    test_states = []
    test_actions = []
    test_goal_info = []
    
    for task_idx, goal_idx in enumerate(test_goal_indices):
        task = mt1.train_tasks[goal_idx]
        goal_success = 0
        
        for ep in range(episodes_per_goal):
            env.set_task(task)
            obs, _ = env.reset()
            done = False
            
            episode_states = []
            episode_actions = []
            
            while not done and len(episode_states) < 500:
                action = policy.get_action(obs)
                episode_states.append(obs)
                episode_actions.append(action)
                
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if info['success']:
                    break
            
            if info['success']:
                test_states.append(np.array(episode_states))
                test_actions.append(np.array(episode_actions))
                test_goal_info.append(goal_idx)
                goal_success += 1
        
        if (task_idx + 1) % 5 == 0 or task_idx == 0:
            pct = (task_idx + 1) / test_goals * 100
            print(f"  Goal {task_idx + 1:2d}/{test_goals}: {goal_success}/{episodes_per_goal} successful | {pct:5.1f}%")
    
    # Save both splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_filename = os.path.join(output_dir, f'expert_data_{task_name}_train.npz')
    test_filename = os.path.join(output_dir, f'expert_data_{task_name}_test.npz')
    
    print(f"\n{'='*70}")
    print(f"SAVING DATA")
    print(f"{'='*70}")
    
    print(f"\nTraining set:")
    print(f"  Trajectories: {len(train_states)}")
    print(f"  Total samples: {sum(len(s) for s in train_states)}")
    print(f"  Saving to: {train_filename}")
    np.savez(train_filename,
             states=np.array(train_states, dtype=object),
             actions=np.array(train_actions, dtype=object),
             goal_indices=train_goal_indices)
    
    print(f"\nTest set:")
    print(f"  Trajectories: {len(test_states)}")
    print(f"  Total samples: {sum(len(s) for s in test_states)}")
    print(f"  Saving to: {test_filename}")
    np.savez(test_filename,
             states=np.array(test_states, dtype=object),
             actions=np.array(test_actions, dtype=object),
             goal_indices=np.array(test_goal_info),
             held_out_goal_indices=test_goal_indices)
    
    print(f"\n✅ Done! Expert data collected with train/test split.\n")
    
    return train_filename, test_filename, train_goal_indices, test_goal_indices


if __name__ == "__main__":
    train_file, test_file, train_goals, test_goals = collect_expert_data_with_split(
        task_name='reach-v3',
        episodes_per_goal=40,
        train_goals=40,
        test_goals=10,
        output_dir='baseline/data'
    )
    
    print("Next steps:")
    print(f"1. Train on: {train_file}")
    print(f"2. Test on held-out goals only (in test file)")
    print(f"   Training goal indices: {list(train_goals[:5])}... (40 total)")
    print(f"   Test goal indices: {list(test_goals)} (10 total)")
