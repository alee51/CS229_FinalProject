import metaworld
import torch
import numpy as np
import os
from train import ClonePolicy

MODEL_PATH = os.path.join('..','models','balanced_cloned_policy.pth')
EPISODES_PER_TASK = 10
TASK_NAME = 'reach-v3'

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    ClonePolicyLocal = ClonePolicy
    model = ClonePolicyLocal(39, 4)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    mt1 = metaworld.MT1(TASK_NAME)
    env_cls = mt1.train_classes[TASK_NAME]

    results = []

    for idx, task in enumerate(mt1.train_tasks):
        env = env_cls()
        success_count = 0
        for ep in range(EPISODES_PER_TASK):
            env.set_task(task)
            try:
                obs, info = env.reset()
            except Exception:
                obs = env.reset()
                info = {}
            done = False
            steps = 0
            while not done and steps < 500:
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    action = model(obs_tensor).numpy()
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except Exception:
                    # older gym signature
                    obs, reward, done, info = env.step(action)
                    terminated = done
                    truncated = False
                done = terminated or truncated
                steps += 1
            succ = bool(info.get('success', False))
            if succ:
                success_count += 1
        rate = success_count / EPISODES_PER_TASK * 100
        results.append((idx, success_count, EPISODES_PER_TASK, rate))
        print(f"Task {idx:02d}: {success_count}/{EPISODES_PER_TASK} -> {rate:.1f}%")

    # Summary
    rates = [r for (_,_,_,r) in results]
    print('\nSummary:')
    print(f'  Mean per-task success: {np.mean(rates):.1f}%')
    print(f'  Median: {np.median(rates):.1f}%')
    print(f'  Min: {np.min(rates):.1f}%')
    print(f'  Max: {np.max(rates):.1f}%')

    perfect = [idx for (idx, s, t, r) in results if s==t]
    zero = [idx for (idx, s, t, r) in results if s==0]
    print(f'\nTasks with 100% success: {len(perfect)} -> {perfect}')
    print(f'Tasks with 0% success: {len(zero)} -> {zero}')

    # Save results
    out_path = os.path.join('..','models','per_task_eval.npy')
    np.save(out_path, np.array(results, dtype=object))
    print(f'\nSaved per-task results to {out_path}')

if __name__ == '__main__':
    main()
