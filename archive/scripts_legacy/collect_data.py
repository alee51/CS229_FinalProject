import metaworld
import numpy as np
import random
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

def collect_expert_trajectories(task_name='reach-v3', num_episodes=2000):
    print(f"🚀 Starting collaborative data collection for {task_name}...")
    
    # 1. Setup the Environment
    # MT1 (Multi-Task 1) creates an environment for a single specific task
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    
    # 2. Load the Expert Bot (The "Teacher")
    # This bot uses internal math to calculate the perfect move
    policy = SawyerReachV3Policy()
    
    # Storage lists
    all_states = []
    all_actions = []
    success_count = 0

    # 3. Collection Loop
    for episode in range(num_episodes):
        # Pick a random starting position/goal from the benchmark list
        task = random.choice(mt1.train_tasks)
        env.set_task(task)
        
        obs, info = env.reset()
        done = False
        steps = 0
        
        # Temp lists for just this one episode
        episode_states = []
        episode_actions = []
        
        while not done and steps < 500:
            # Ask the expert bot what to do
            # The policy needs the current observation (obs) to decide
            action = policy.get_action(obs)
            
            # Record the data
            episode_states.append(obs)
            episode_actions.append(action)
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Check if we succeeded
            if info['success']:
                success_count += 1
                break 

        # Only keep the data if the expert actually succeeded
        if info['success']:
            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))
            if (episode + 1) % 100 == 0:
                print(f"✅ Collected {episode + 1} / {num_episodes} episodes...")
        else:
            print(f"❌ Episode {episode+1}: Expert failed to reach target.")

    # 4. Save to file
    # Saving as a dictionary of numpy arrays is standard for PyTorch dataloaders
    filename = f'expert_data_{task_name}.npz'
    print(f"\n📦 Saving {len(all_states)} successful trajectories to {filename}...")
    np.savez(filename, 
             states=np.array(all_states, dtype=object), 
             actions=np.array(all_actions, dtype=object))
    print("Done! Team data ready.")

if __name__ == "__main__":
    collect_expert_trajectories()