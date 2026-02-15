import metaworld
import torch
import numpy as np
import sys
sys.path.append('.')
from train_variants import ClonePolicy

def test_policy(model_path, num_episodes=1000, clip_actions=False):
    """Test a single policy"""
    
    mt1 = metaworld.MT1('reach-v3')
    env = mt1.train_classes['reach-v3']()
    
    model = ClonePolicy(39, 4)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"❌ Failed to load {model_path}: {e}")
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
            
            # Clip if this model was trained with clipping
            if clip_actions:
                action = np.clip(action, -1.0, 1.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info['success']:
            success_count += 1

    success_rate = success_count/num_episodes*100
    return success_rate

if __name__ == "__main__":
    models = [
        'baseline_original.pth',
        'baseline_lr001_e50.pth',
        'baseline_lr005_e50.pth',
        'baseline_larger_e50.pth',
        'baseline_lr001_b32_e50.pth',
        'baseline_lr001_e50_CLIPPED.pth',
    ]
    
    # Specify which models need action clipping during testing
    clip_flags = {
        'baseline_lr001_e50_CLIPPED.pth': True,
    }
    
    print("\nTesting all variants (100 episodes each)...\n")
    results = {}
    
    for model_path in models:
        should_clip = clip_flags.get(model_path, False)
        result = test_policy(model_path, num_episodes=100, clip_actions=should_clip)
        if result is not None:
            clip_txt = " [CLIPPED]" if should_clip else ""
            results[model_path] = result
            print(f"✅ {model_path:35s} -> {result:6.2f}%{clip_txt}")
        else:
            print(f"❌ {model_path:35s} -> FAILED")
    
    print("\n" + "="*60)
    print("SUMMARY (sorted by performance):")
    print("="*60)
    for model, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{rate:6.2f}%  {model}")
