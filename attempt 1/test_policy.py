import metaworld
import torch
import torch.nn as nn
import numpy as np
import time

# --- 🛠️ SETTINGS 🛠️ ---
VISUALIZE = False       # Set to True to watch, False to run fast
NUM_EPISODES = 50       # How many tests to run
# -----------------------

class ClonePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClonePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased neurons to 256 for more brainpower
            nn.ReLU(),
            nn.Linear(256, 256),        # Added deeper layers
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def test_student_robot(task_name='reach-v3'):
    print(f"🎓 Testing Student Robot (Visual Mode: {VISUALIZE})...")
    
    # Setup Environment
    mt1 = metaworld.MT1(task_name)
    
    if VISUALIZE:
        # Window opens
        env = mt1.train_classes[task_name](render_mode="human")
    else:
        # No window (Headless)
        env = mt1.train_classes[task_name]() 
    
    # Load Brain
    model = ClonePolicy(39, 4)
    # Make sure this matches your file path!
    model.load_state_dict(torch.load('cloned_policy2.pth'))
    model.eval()

    success_count = 0

    try:
        for i in range(NUM_EPISODES):
            # Pick a random task
            task = mt1.train_tasks[i % len(mt1.train_tasks)]
            env.set_task(task)
            obs, info = env.reset()
            
            if VISUALIZE: env.render()
            
            done = False
            steps = 0
            
            while not done and steps < 500:
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    action = model(obs_tensor).numpy()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if VISUALIZE:
                    env.render()
                    time.sleep(0.01) # Only sleep if we are watching!
                
                done = terminated or truncated
                steps += 1

            if info['success']:
                success_count += 1
                # Optional: Print every success if visual
                if VISUALIZE: print(f"Episode {i+1}: ✅ Success")
            else:
                if VISUALIZE: print(f"Episode {i+1}: ❌ Failed")

        # Final Report
        accuracy = (success_count / NUM_EPISODES) * 100
        print(f"\n📊 RESULTS: {success_count}/{NUM_EPISODES} Successful")
        print(f"🏆 Success Rate: {accuracy:.1f}%")

    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    test_student_robot()