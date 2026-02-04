import numpy as np

def check_if_ball_moves():
    print("🕵️ INSPECTING DATA...")
    try:
        # Load your dataset
        data = np.load('expert_data_reach-v3.npz', allow_pickle=True)
        states = data['states'] # This is a list of arrays
    except:
        print("❌ Data file not found.")
        return

    print(f"Total Episodes Collected: {len(states)}")
    
    # In Meta-World 'reach-v3', the Goal Position (Ball) is usually 
    # the LAST 3 numbers of the observation array.
    # Let's verify if they change between episodes.
    
    first_episode_ball = states[0][0][-3:] # First frame of first episode
    print(f"\nEpisode 1 Ball Location: {first_episode_ball}")
    
    diff_count = 0
    
    for i in range(1, min(5, len(states))):
        this_episode_ball = states[i][0][-3:] # First frame of episode 'i'
        print(f"Episode {i+1} Ball Location: {this_episode_ball}")
        
        # Check if it's different from the first one
        if not np.allclose(first_episode_ball, this_episode_ball):
            diff_count += 1

    print("\n--- DIAGNOSIS ---")
    if diff_count > 0:
        print("✅ GOOD: The ball IS moving to different spots.")
        print("   The robot CAN learn relative distance.")
        print("   -> Solution: You just need MORE data (2000 episodes).")
    else:
        print("⚠️ BAD: The ball is ALWAYS in the same spot.")
        print("   The robot CANNOT learn.")
        print("   -> Solution: We need to fix the 'collect_data.py' script.")

if __name__ == "__main__":
    check_if_ball_moves()