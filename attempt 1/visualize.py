import metaworld
import random
import time
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

def watch_robot_fixed(task_name='reach-v3'):
    print(f"🍿 Opening Visualization for {task_name}")
    print("👉 Press 'Ctrl + C' in the terminal to stop.")
    
    # 1. Enable Human Render Mode
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name](render_mode="human")
    policy = SawyerReachV3Policy()

    try:
        while True:
            # Pick a new random position
            task = random.choice(mt1.train_tasks)
            env.set_task(task)
            
            obs, info = env.reset()
            done = False
            steps = 0
            
            # 2. Critical Fix: Render once before the loop starts
            env.render()
            
            while not done and steps < 500:
                action = policy.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 3. CRITICAL FIX: Force the window to update!
                # Without this line, Windows thinks the app is dead.
                env.render()
                
                done = terminated or truncated
                steps += 1
                
                # Keep the sleep short so the window doesn't lag
                time.sleep(0.02) 

            if info['success']:
                print("✅ Target Reached!")
            else:
                print("❌ Failed.")
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Closing...")
        env.close()

if __name__ == "__main__":
    watch_robot_fixed()