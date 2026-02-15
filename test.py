#!/usr/bin/env python
"""
Unified test script for CS229 project policies

Usage:
    python test.py --approach baseline --model cloned_policy.pth --episodes 50
    python test.py --approach baseline --model cloned_policy.pth --episodes 1 --clip --visualize
    python test.py --approach baseline --model cloned_policy.pth --clip --visualize-series 5
    python test.py --approach baseline --model cloned_policy.pth --clip --visualize-success-fail 3  # 3 success + 3 fail
    python test.py --approach baseline --model cloned_policy.pth --clip --visualize-parallel 5
"""

import metaworld
import torch
import numpy as np
import argparse
import sys
import os

def load_model_class(approach):
    """Dynamically load the ClonePolicy from the appropriate approach"""
    if approach == 'baseline':
        sys.path.insert(0, os.path.join('baseline', 'scripts'))
    elif approach in ['vae', 'tce', 'hybrid']:
        sys.path.insert(0, os.path.join(approach, 'scripts'))
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    # Try to import the train module which contains the model class
    try:
        from train import ClonePolicy
        return ClonePolicy
    except ImportError:
        print(f"Could not import ClonePolicy from {approach}/scripts/train.py")
        sys.exit(1)

def test_policy(approach, model_name, num_episodes=50, task_name='reach-v3', 
                clip_actions=False, verbose=0, visualize=False, goal_indices=None,
                return_episode_results=False):
    """Test a policy model from a specific approach.
    
    Args:
        approach: 'baseline', 'vae', 'tce', or 'hybrid'
        model_name: Name of the model file (e.g., cloned_policy.pth)
        num_episodes: Number of episodes to test
        task_name: MetaWorld task name
        clip_actions: Whether to clip actions to [-1, 1] (recommended; env expects [-1,1])
        verbose: Print progress every N episodes (0 = off)
        visualize: If True, render env and sleep each step so you can watch.
        goal_indices: If set, run only these goal indices (in order); len must match num_episodes.
        return_episode_results: If True, return (success_rate, [(goal_idx, success), ...]).
    
    Returns:
        success_rate, or (success_rate, episode_results) if return_episode_results.
    """
    import time
    
    model_path = os.path.join(approach, 'models', model_name)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    ClonePolicy = load_model_class(approach)
    try:
        mt1 = metaworld.MT1(task_name)
        env_cls = mt1.train_classes[task_name]
        env = env_cls(render_mode="human") if visualize else env_cls()
    except TypeError:
        mt1 = metaworld.MT1(task_name)
        env_cls = mt1.train_classes[task_name]
        env = env_cls()
        if visualize:
            print("Warning: render_mode not supported by this env; running without visualization.")
    
    try:
        model = ClonePolicy(39, 4)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception as e:
        print(f"Failed to load model '{model_path}': {e}")
        return None
    
    model.eval()
    success_count = 0
    episode_results = []  # (goal_idx, success)
    
    for i in range(num_episodes):
        goal_idx = goal_indices[i] if goal_indices is not None else (i % len(mt1.train_tasks))
        task = mt1.train_tasks[goal_idx]
        env.set_task(task)
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) >= 2:
            obs, info = reset_out[0], reset_out[1] if len(reset_out) > 1 else {}
        else:
            obs, info = reset_out, {}
        done = False
        
        while not done:
            obs_flat = np.asarray(obs).flatten()
            obs_tensor = torch.FloatTensor(obs_flat)
            with torch.no_grad():
                action = model(obs_tensor).numpy()
            action = np.asarray(action).flatten().astype(np.float64)
            if clip_actions:
                action = np.clip(action, -1.0, 1.0)
            
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
            else:
                obs, reward, done, info = step_out
                terminated, truncated = done, False
            done = terminated or truncated
            
            if visualize:
                try:
                    env.render()
                except Exception:
                    pass
                time.sleep(0.02)

        succ = info.get('success', False)
        if succ:
            success_count += 1
        episode_results.append((goal_idx, succ))
        
        if verbose and (i + 1) % verbose == 0:
            print(f"  Progress: {i+1}/{num_episodes} | Current rate: {success_count/(i+1)*100:.1f}%")
        if visualize:
            print(f"  Episode {i+1} (goal {goal_idx}): {'success' if succ else 'fail'}")

    if visualize:
        try:
            env.close()
        except Exception:
            pass
    success_rate = success_count / num_episodes * 100
    if return_episode_results:
        return success_rate, episode_results
    return success_rate


def visualize_success_fail(approach, model_name, n_each=3, task_name='reach-v3', clip_actions=False):
    """Run a quick eval (no render), then visualize n_each successes and n_each failures in series."""
    print("Running quick eval on all 50 goals (no render)...")
    out = test_policy(approach, model_name, num_episodes=50, task_name=task_name,
                      clip_actions=clip_actions, verbose=0, visualize=False,
                      return_episode_results=True)
    if out is None:
        return None
    rate, episode_results = out
    success_goals = [g for g, s in episode_results if s]
    fail_goals = [g for g, s in episode_results if not s]
    n_s = min(n_each, len(success_goals))
    n_f = min(n_each, len(fail_goals))
    if n_s == 0 and n_f == 0:
        print("No episodes to show.")
        return rate
    goals_to_show = success_goals[:n_s] + fail_goals[:n_f]
    print(f"Overall: {rate:.1f}% | Showing {n_s} success + {n_f} fail (goals {goals_to_show})")
    print("Opening visualization (same window, in order)...\n")
    test_policy(approach, model_name, num_episodes=len(goals_to_show), task_name=task_name,
                clip_actions=clip_actions, verbose=0, visualize=True, goal_indices=goals_to_show)
    return rate


def test_policy_parallel_visualize(approach, model_name, n_parallel=5, task_name='reach-v3',
                                   clip_actions=False):
    """Run n_parallel envs at once, each with a different goal, all rendering so you can compare.
    Steps all envs in lockstep; each window shows one goal. Good for spotting shared failure modes.
    """
    import time

    model_path = os.path.join(approach, 'models', model_name)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    ClonePolicy = load_model_class(approach)
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]

    try:
        envs = [env_cls(render_mode="human") for _ in range(n_parallel)]
    except TypeError:
        print("Warning: render_mode not supported; cannot run parallel visualize.")
        return None

    try:
        model = ClonePolicy(39, 4)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception as e:
        print(f"Failed to load model: {e}")
        for e in envs:
            e.close()
        return None

    model.eval()
    tasks = mt1.train_tasks
    n_parallel = min(n_parallel, len(tasks))

    for i in range(n_parallel):
        envs[i].set_task(tasks[i])
    reset_out = [envs[i].reset() for i in range(n_parallel)]
    obs_list = []
    info_list = []
    for i in range(n_parallel):
        out = reset_out[i]
        if isinstance(out, tuple) and len(out) >= 2:
            obs_list.append(np.asarray(out[0]).flatten())
            info_list.append(out[1] if len(out) > 1 else {})
        else:
            obs_list.append(np.asarray(out).flatten())
            info_list.append({})

    done_list = [False] * n_parallel
    step_count = 0
    max_steps = 500

    while not all(done_list) and step_count < max_steps:
        actions = []
        for i in range(n_parallel):
            if done_list[i]:
                actions.append(None)
                continue
            obs_t = torch.FloatTensor(obs_list[i])
            with torch.no_grad():
                a = model(obs_t).numpy()
            a = np.asarray(a).flatten().astype(np.float64)
            if clip_actions:
                a = np.clip(a, -1.0, 1.0)
            actions.append(a)

        for i in range(n_parallel):
            if done_list[i] or actions[i] is None:
                continue
            step_out = envs[i].step(actions[i])
            if len(step_out) == 5:
                obs_list[i], _, term, trunc, info_list[i] = step_out
            else:
                obs_list[i], _, d, info_list[i] = step_out
                term, trunc = d, False
            done_list[i] = term or trunc
            if not done_list[i]:
                obs_list[i] = np.asarray(obs_list[i]).flatten()

        for i in range(n_parallel):
            try:
                envs[i].render()
            except Exception:
                pass
        time.sleep(0.02)
        step_count += 1

    success_count = sum(1 for i in range(n_parallel) if info_list[i].get('success', False))
    print(f"\nParallel visualize ({n_parallel} goals):")
    for i in range(n_parallel):
        print(f"  Goal {i}: {'success' if info_list[i].get('success') else 'fail'}")
    print(f"  Total: {success_count}/{n_parallel}")

    for e in envs:
        try:
            e.close()
        except Exception:
            pass

    return success_count / n_parallel * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test policies for CS229 project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --approach baseline --model cloned_policy_stable2.pth
  python test.py --approach baseline --model cloned_policy.pth --episodes 500 --verbose 50
  python test.py --approach baseline --model baseline_lr001_e50.pth --clip
        """
    )
    
    parser.add_argument('--approach', type=str, default='baseline', 
                        choices=['baseline', 'vae', 'tce', 'hybrid'],
                        help='Which approach to test (default: baseline)')
    parser.add_argument('--model', type=str, required=True, 
                        help='Model filename (e.g., cloned_policy_stable2.pth)')
    parser.add_argument('--episodes', type=int, default=50, 
                        help='Number of test episodes (default: 50, one per goal)')
    parser.add_argument('--task', type=str, default='reach-v3', 
                        help='MetaWorld task name (default: reach-v3)')
    parser.add_argument('--clip', action='store_true', 
                        help='Clip actions to [-1, 1]')
    parser.add_argument('--verbose', type=int, default=0, 
                        help='Print progress every N episodes (0=off)')
    parser.add_argument('--visualize', action='store_true',
                        help='Render env and run slowly so you can watch (use with --episodes 1)')
    parser.add_argument('--visualize-parallel', type=int, default=0, metavar='N',
                        help='Open N envs at once with different goals; step in sync (e.g. 5 for 5 windows)')
    parser.add_argument('--visualize-series', type=int, default=0, metavar='N',
                        help='Same window: run N episodes one after another (e.g. 5 = goals 0..4 in series)')
    parser.add_argument('--visualize-success-fail', type=int, default=0, metavar='N',
                        help='Eval 50 goals, then show N successes + N failures in same window (e.g. 3)')
    
    args = parser.parse_args()
    
    if args.visualize_success_fail > 0:
        print(f"\n{'='*70}")
        print(f"Visualize N success + N fail (N={args.visualize_success_fail})")
        print(f"{'='*70}")
        print(f"Model: {args.model}  Clip: {'Yes' if args.clip else 'No'}\n")
        result = visualize_success_fail(
            approach=args.approach,
            model_name=args.model,
            n_each=args.visualize_success_fail,
            task_name=args.task,
            clip_actions=args.clip
        )
        if result is not None:
            print(f"\nOverall success rate: {result:.2f}%")
        sys.exit(0 if result is not None else 1)
    
    if args.visualize_series > 0:
        args.visualize = True
        args.episodes = args.visualize_series
    
    if args.visualize_parallel > 0:
        print(f"\n{'='*70}")
        print(f"Parallel visualize: {args.visualize_parallel} envs (different goals)")
        print(f"{'='*70}")
        print(f"Model: {args.model}  Clip: {'Yes' if args.clip else 'No'}\n")
        result = test_policy_parallel_visualize(
            approach=args.approach,
            model_name=args.model,
            n_parallel=args.visualize_parallel,
            task_name=args.task,
            clip_actions=args.clip
        )
        if result is not None:
            print(f"\nSUCCESS RATE (this run): {result:.2f}%")
        sys.exit(0 if result is not None else 1)
    
    if args.visualize and args.episodes > 5 and args.visualize_series == 0:
        print("(Visualize mode: limiting to 5 episodes. Use --visualize-series N for more in same window.)")
        args.episodes = min(args.episodes, 5)
    
    print(f"\n{'='*70}")
    print(f"Testing Policy")
    print(f"{'='*70}")
    print(f"Approach:        {args.approach}")
    print(f"Model:           {args.model}")
    print(f"Task:            {args.task}")
    print(f"Episodes:        {args.episodes}")
    print(f"Clip Actions:    {'Yes' if args.clip else 'No'}")
    print(f"Visualize:       {'Yes' if args.visualize else 'No'}")
    print(f"{'='*70}\n")
    
    result = test_policy(
        approach=args.approach,
        model_name=args.model,
        num_episodes=args.episodes,
        task_name=args.task,
        clip_actions=args.clip,
        verbose=args.verbose,
        visualize=args.visualize
    )
    
    if result is not None:
        print(f"\n{'='*70}")
        print(f"SUCCESS RATE: {result:.2f}%")
        print(f"{'='*70}\n")
    else:
        print(f"\nTest failed\n")
