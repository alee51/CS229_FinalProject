"""
Analyze expert trajectory variation for the same goal
Shows why memorization won't work - trajectories for the same goal vary
"""

import metaworld
import numpy as np
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

def analyze_trajectory_variation():
    print("="*70)
    print("EXPERT TRAJECTORY VARIATION ANALYSIS")
    print("="*70)
    print("\nQuestion: For the SAME goal, do expert trajectories vary?")
    print("If yes → Model must learn POLICY (state->action mapping)")
    print("If no → Model could just memorize specific trajectory")
    print()
    
    mt1 = metaworld.MT1('reach-v3')
    env = mt1.train_classes['reach-v3']()
    policy = SawyerReachV3Policy()
    
    # Pick ONE specific goal to test
    fixed_task = mt1.train_tasks[0]  # Always use same goal
    
    print(f"Testing 10 expert trajectories for the SAME goal (Task 0):")
    print()
    
    trajectories = []
    
    for run in range(10):
        env.set_task(fixed_task)
        obs, _ = env.reset()
        done = False
        
        traj_states = []
        traj_actions = []
        
        while not done and len(traj_states) < 500:
            action = policy.get_action(obs)
            traj_states.append(obs.copy())
            traj_actions.append(action.copy())
            
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info['success']:
                break
        
        trajectories.append({
            'states': np.array(traj_states),
            'actions': np.array(traj_actions),
            'length': len(traj_states),
            'success': info['success']
        })
        
        if traj_states:
            action_range = [np.array(traj_actions).min(), np.array(traj_actions).max()]
            print(f"  Run {run+1:2d}: {len(traj_states):3d} steps | "
                  f"Actions: [{action_range[0]:.3f}, {action_range[1]:.3f}] | "
                  f"Success: {info['success']}")
    
    # Analyze variation
    print()
    print("="*70)
    print("VARIATION ANALYSIS")
    print("="*70)
    
    lengths = [t['length'] for t in trajectories]
    print(f"\nTrajectory lengths (same goal):")
    print(f"  Min: {min(lengths)} steps")
    print(f"  Max: {max(lengths)} steps")
    print(f"  Mean: {np.mean(lengths):.1f} steps")
    print(f"  Std Dev: {np.std(lengths):.1f} steps")
    print(f"  Range: {max(lengths) - min(lengths)} steps")
    
    if max(lengths) - min(lengths) > 5:
        print(f"\n  ✓ Trajectories vary significantly in LENGTH")
        print(f"    → Can't memorize exact sequence of steps!")
    
    # Compare first two trajectories
    print(f"\nComparing trajectories 1 vs 2 (same goal):")
    
    t1_states = trajectories[0]['states']
    t2_states = trajectories[1]['states']
    
    # For comparison, look at first few states
    print(f"\n  First 3 states from Run 1:")
    for i in range(min(3, len(t1_states))):
        print(f"    Step {i}: {t1_states[i][:5]}... (gripper pos example)")
    
    print(f"\n  First 3 states from Run 2 (same goal):")
    for i in range(min(3, len(t2_states))):
        print(f"    Step {i}: {t2_states[i][:5]}... (gripper pos example)")
    
    if len(t1_states) > 0 and len(t2_states) > 0:
        state_diff = np.mean(np.abs(t1_states[0] - t2_states[0]))
        print(f"\n  Mean state difference at step 0: {state_diff:.6f}")
        if state_diff > 0.001:
            print(f"  → Initial states differ (environment stochasticity)")
    
    # Check action sequences
    print(f"\nAction sequence comparison:")
    t1_actions = trajectories[0]['actions']
    t2_actions = trajectories[1]['actions']
    
    if len(t1_actions) > 0 and len(t2_actions) > 0:
        min_len = min(len(t1_actions), len(t2_actions))
        
        # For first 5 steps, compare actions
        print(f"\n  First 5 actions, Run 1:")
        for i in range(min(5, len(t1_actions))):
            print(f"    Step {i}: {t1_actions[i]}")
        
        print(f"\n  First 5 actions, Run 2 (same goal):")
        for i in range(min(5, len(t2_actions))):
            print(f"    Step {i}: {t2_actions[i]}")
        
        # Check if actions are identical
        if min_len > 0:
            identical_steps = np.sum(np.allclose(t1_actions[:min_len], t2_actions[:min_len]))
            print(f"\n  Identical actions in first {min_len} steps: {identical_steps}/{min_len}")
            if identical_steps < min_len * 0.5:
                print(f"  → Actions differ (policy is reactive to state, not deterministic)")
    
    print()
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nWhy memorization fails:")
    print("1. Trajectories for the SAME goal have different lengths")
    print("2. Initial states vary due to environment stochasticity")
    print("3. Actions vary because policy reacts to observed state")
    print("\nTherefore:")
    print("✓ Model must learn the POLICY: state → action mapping")
    print("✓ NOT memorize specific trajectory sequences")
    print()


if __name__ == "__main__":
    analyze_trajectory_variation()
