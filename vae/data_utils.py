import metaworld
import numpy as np
import torch
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

def collect_expert_data(task_name="reach-v3", num_episodes=100):
    """
    Generates expert demonstrations using Meta-World's scripted policies.
    """
    # 1. Setup the MT1 benchmark for the specific task
    mt1 = metaworld.MT1(task_name) 
    env = mt1.train_classes[task_name]()
    task = mt1.train_tasks[0]
    env.set_task(task)

    # 2. Select the correct scripted expert policy
    if "reach" in task_name:
        expert = SawyerReachV3Policy()
    # elif "push" in task_name:
    #     expert = SawyerPushV3Policy()
    else:
        raise ValueError("Policy for this task is not defined in this script.")

    all_obs = []
    all_actions = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Scripted policy generates the 'expert' action
            action = expert.get_action(obs)
            
            all_obs.append(obs)
            all_actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return np.array(all_obs), np.array(all_actions)