#!/usr/bin/env python
"""
Unified test script for CS229 project policies.
Single entrypoint: use this script for all testing (baseline and other approaches).
Baseline eval logic lives in baseline/scripts/test.py; root delegates to it when --approach baseline.

Usage:
    python test.py --approach baseline --model cloned_policy.pth --episodes 50
    python test.py --approach baseline --model cloned_policy.pth --suite mt10
    python test.py --approach baseline --model cloned_policy.pth --episodes 1 --visualize
    python test.py --approach baseline --model cloned_policy.pth --visualize-series 5
    python test.py --approach baseline --model cloned_policy.pth --visualize-success-fail 3
    python test.py --approach baseline --model latest-upsampled-end --visualize-success-fail 3
    python test.py --approach baseline --model cloned_policy.pth --visualize-parallel 5
    # Clipping is on by default (same as train.py). Use --no-clip to disable.
"""

import metaworld
import torch
import numpy as np
import argparse
import sys
import os
import json
import time
import importlib.util

# #region agent log
DEBUG_LOG = os.path.join(os.path.dirname(__file__), ".cursor", "debug.log")
def _dlog(msg, data, hypothesis_id):
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(json.dumps({"timestamp": int(time.time()*1000), "message": msg, "data": data, "hypothesisId": hypothesis_id}) + "\n")
    except Exception:
        pass
# #endregion

def resolve_model_name(approach, model_name):
    """If model_name is 'latest-upsampled-end', resolve to runs/run_<timestamp>.pth from training_runs.json."""
    if model_name not in ('latest-upsampled-end', 'upsampled-end'):
        return model_name
    runs_path = os.path.join(approach, 'training_runs.json')
    if not os.path.exists(runs_path):
        return model_name
    try:
        with open(runs_path, 'r') as f:
            runs_list = json.load(f)
    except Exception:
        return model_name
    upsampled = [r for r in runs_list if r.get('end_weight', 1) != 1]
    if not upsampled:
        return model_name
    ts = upsampled[-1].get('timestamp', '')
    if not ts:
        return model_name
    return os.path.join('runs', f'run_{ts}.pth')

def _load_baseline_test():
    """Load baseline/scripts/test.py as a module (avoid naming conflict with this script)."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root_dir, 'baseline', 'scripts', 'test.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("baseline_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log_test_to_wandb(use_wandb, approach, model, task_or_suite, episodes, seed, clip_actions,
                       result_single=None, result_mt=None, task_list=None, device=None, tags=None):
    """Log a test run to W&B (project cs229-metaworld, job_type=eval). No-op if use_wandb is False or no result."""
    if not use_wandb:
        return
    if result_single is None and result_mt is None:
        return
    try:
        import wandb
        model_stem = os.path.splitext(os.path.basename(model))[0]
        run_name = f"eval-{task_or_suite}-{model_stem}-{episodes}ep"
        run = wandb.init(project="cs229-metaworld", job_type="eval", name=run_name, tags=tags or [], reinit=True)
        wandb.config.update({
            "approach": approach,
            "model": os.path.basename(model),
            "task_or_suite": task_or_suite,
            "episodes": episodes,
            "seed": seed,
            "clip_actions": clip_actions,
        }, allow_val_change=True)
        if device is not None:
            wandb.config.update({"device": str(device)}, allow_val_change=True)
        if result_single is not None:
            wandb.log({"eval/success_rate": result_single})
        elif result_mt is not None:
            success_per_task, avg = result_mt
            wandb.log({"eval/success_rate_avg": avg})
            if task_list is not None and len(task_list) == len(success_per_task):
                for i, r in enumerate(success_per_task):
                    wandb.log({f"eval/success_rate_{task_list[i]}": r})
        wandb.finish()
    except Exception as e:
        print(f"W&B logging skipped: {e}")


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
                clip_actions=True, verbose=0, visualize=False, goal_indices=None,
                return_episode_results=False, eval_seed=None, visualize_n_after_50=None):
    """Test a policy model from a specific approach.
    
    Args:
        approach: 'baseline', 'vae', 'tce', or 'hybrid'
        model_name: Name of the model file (e.g., cloned_policy.pth)
        num_episodes: Number of episodes to test
        task_name: MetaWorld task name
        clip_actions: Whether to clip actions to [-1, 1] (default True; matches train.py)
        verbose: Print progress every N episodes (0 = off)
        visualize: If True, render env and sleep each step so you can watch.
        goal_indices: If set, run only these goal indices (in order); len must match num_episodes.
        return_episode_results: If True, return (success_rate, [(goal_idx, success), ...]).
        eval_seed: If set, pass seed to env.reset(seed=eval_seed+goal_idx) for deterministic eval.
    
    Returns:
        success_rate, or (success_rate, episode_results) if return_episode_results.
    """
    import time

    # When eval_seed is set, fix NumPy and PyTorch RNG so the whole eval is reproducible.
    if eval_seed is not None:
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    model_path = os.path.join(approach, 'models', model_name)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    ClonePolicy = load_model_class(approach)
    will_render = visualize or (visualize_n_after_50 is not None)
    try:
        mt1 = metaworld.MT1(task_name)
        env_cls = mt1.train_classes[task_name]
        env = env_cls(render_mode="human") if will_render else env_cls()
    except TypeError:
        mt1 = metaworld.MT1(task_name)
        env_cls = mt1.train_classes[task_name]
        env = env_cls()
        if will_render:
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
    step_counts = []  # steps per episode (for debug)
    
    for i in range(num_episodes):
        goal_idx = goal_indices[i] if goal_indices is not None else (i % len(mt1.train_tasks))
        task = mt1.train_tasks[goal_idx]
        env.set_task(task)
        if eval_seed is not None:
            try:
                reset_out = env.reset(seed=eval_seed + goal_idx)
            except TypeError:
                reset_out = env.reset()
        else:
            reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) >= 2:
            obs, info = reset_out[0], reset_out[1] if len(reset_out) > 1 else {}
        else:
            obs, info = reset_out, {}
        done = False
        steps = 0
        max_steps = 500  # match train eval_50_goals
        while not done and steps < max_steps:
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
            steps += 1
            
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
        step_counts.append(steps)
        
        if verbose and (i + 1) % verbose == 0:
            print(f"  Progress: {i+1}/{num_episodes} | Current rate: {success_count/(i+1)*100:.1f}%")
        if visualize:
            print(f"  Episode {i+1} (goal {goal_idx}): {'success' if succ else 'fail'}")

    # Phase 2: same env, run first n_s success + n_f fail with render (labels will match) same env, run first n_s success + n_f fail with render (labels will match)
    if visualize_n_after_50 is not None and num_episodes == 50 and len(episode_results) == 50:
        n_s, n_f = visualize_n_after_50
        success_goals = [g for g, s in episode_results if s]
        fail_goals = [g for g, s in episode_results if not s]
        n_s = min(n_s, len(success_goals))
        n_f = min(n_f, len(fail_goals))
        if n_s or n_f:
            goals_to_show = success_goals[:n_s] + fail_goals[:n_f]
            rate_50 = success_count / 50 * 100
            print(f"\nOverall (50 goals): {rate_50:.1f}%")
            print(f"Showing {n_s} SUCCESS (goals {goals_to_show[:n_s]}) then {n_f} FAIL (goals {goals_to_show[n_s:]}). Same env — labels match.")
            print("(Labels use the env's binary success threshold; what you see may look better or worse than the env result.)\n")
            # #region agent log
            _dlog("Phase2 goals_to_show", {"goals_to_show": goals_to_show, "n_s": n_s, "n_f": n_f}, "order")
            for g in goals_to_show:
                idx_in_50 = next((i for i in range(50) if episode_results[i][0] == g), None)
                phase1_steps = step_counts[idx_in_50] if idx_in_50 is not None else None
                phase1_succ = episode_results[idx_in_50][1] if idx_in_50 is not None else None
                _dlog("Phase1 result for goal", {"goal": g, "phase1_success": phase1_succ, "phase1_steps": phase1_steps}, "phase1_vs_phase2")
            # #endregion
            for j in range(len(goals_to_show)):
                goal_idx = goals_to_show[j]
                is_success = j < n_s
                print(f"  >>> Episode {j+1}/{len(goals_to_show)}: Goal {goal_idx} — {'SUCCESS' if is_success else 'FAIL'} <<<")
                task = mt1.train_tasks[goal_idx]
                env.set_task(task)
                if eval_seed is not None:
                    try:
                        reset_out = env.reset(seed=eval_seed + goal_idx)
                    except TypeError:
                        reset_out = env.reset()
                else:
                    reset_out = env.reset()
                if isinstance(reset_out, tuple) and len(reset_out) >= 2:
                    obs, info = reset_out[0], reset_out[1] if len(reset_out) > 1 else {}
                else:
                    obs, info = reset_out, {}
                done = False
                steps_phase2 = 0
                while not done:
                    obs_flat = np.asarray(obs).flatten()
                    with torch.no_grad():
                        action = model(torch.FloatTensor(obs_flat)).numpy()
                    action = np.asarray(action).flatten().astype(np.float64)
                    if clip_actions:
                        action = np.clip(action, -1.0, 1.0)
                    step_out = env.step(action)
                    if len(step_out) == 5:
                        obs, _, term, trunc, info = step_out
                    else:
                        obs, _, done, info = step_out
                        term, trunc = done, False
                    done = term or trunc
                    steps_phase2 += 1
                    try:
                        env.render()
                    except Exception:
                        pass
                    time.sleep(0.02)
                succ = info.get('success', False)
                print(f"  -> Env result: {'success' if succ else 'fail'}")
                # #region agent log
                idx_in_50 = next((i for i in range(50) if episode_results[i][0] == goal_idx), None)
                phase1_steps = step_counts[idx_in_50] if idx_in_50 is not None else None
                _dlog("Phase2 episode", {"j": j, "goal_idx": goal_idx, "label_success": is_success, "actual_success": succ, "phase2_steps": steps_phase2, "phase1_steps": phase1_steps}, "label_vs_actual")
                # #endregion

    if will_render:
        try:
            env.close()
        except Exception:
            pass
    success_rate = success_count / num_episodes * 100
    if return_episode_results:
        return success_rate, episode_results
    return success_rate


def visualize_success_fail(approach, model_name, n_each=3, task_name='reach-v3', clip_actions=True):
    """One run: 50 episodes (no render), then 3 success + 3 fail with render. Same env so labels match."""
    print("Eval 50 goals (no render), then show 3 success + 3 fail in same run.\n")
    out = test_policy(approach, model_name, num_episodes=50, task_name=task_name,
                      clip_actions=clip_actions, verbose=0, visualize=False,
                      return_episode_results=True, eval_seed=42,
                      visualize_n_after_50=(n_each, n_each))
    if out is None:
        return None
    rate, _ = out
    return rate


def test_policy_parallel_visualize(approach, model_name, n_parallel=5, task_name='reach-v3',
                                   clip_actions=True):
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
  python test.py --approach baseline --model baseline_lr001_e50.pth
  python test.py --approach baseline --model my.pth --no-clip   # disable action clipping
        """
    )
    
    parser.add_argument('--approach', type=str, default='baseline', 
                        choices=['baseline', 'vae', 'tce', 'hybrid'],
                        help='Which approach to test (default: baseline)')
    parser.add_argument('--model', type=str, required=True, 
                        help='Model filename (e.g. latest.pth), or latest-upsampled-end for latest run with end_weight!=1')
    parser.add_argument('--episodes', type=int, default=50, 
                        help='Number of test episodes (default: 50, one per goal)')
    parser.add_argument('--task', type=str, default='reach-v3', 
                        help='MetaWorld task name (default: reach-v3)')
    parser.add_argument('--no-clip', action='store_true', 
                        help='Do not clip actions (default: clip to [-1, 1], same as train.py)')
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
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='Seed for env.reset(seed=seed+goal_idx) for reproducible eval; omit for stochastic')
    parser.add_argument("--suite", type=str, default="mt1", choices=["mt1", "mt10", "mt50"],
                        help="mt1: single task (default). mt10/mt50: multi-task, 50 goals per task (baseline only)")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'xpu', 'cpu'],
                        help='Device for baseline (default: auto); ignored for other approaches')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging for this test run')
    parser.add_argument('--wandb-tag', type=str, action='append', default=None, metavar='KEY:VALUE',
                        help='Tag for W&B (repeatable, e.g. --wandb-tag model:mt10-500ep)')
    
    args = parser.parse_args()
    args.model = resolve_model_name(args.approach, args.model)

    # Option A: single entrypoint — delegate all baseline testing to baseline/scripts/test.py
    if args.approach == 'baseline':
        baseline_eval = _load_baseline_test()
        if baseline_eval is None:
            print("Could not load baseline/scripts/test.py")
            sys.exit(1)
        full_model_path = os.path.join(args.approach, 'models', args.model)
        if not os.path.exists(full_model_path):
            print(f"Model not found: {full_model_path}")
            sys.exit(1)
        device = baseline_eval.get_device(args.device)

        if args.task not in baseline_eval.MT10_TASKS:
            print(f"Task '{args.task}' not found. Valid tasks: {', '.join(baseline_eval.MT10_TASKS)}.")
            sys.exit(1)

        if args.visualize_success_fail > 0:
            print(f"\n{'='*70}")
            print(f"Visualize N success + N fail (N={args.visualize_success_fail})")
            print(f"{'='*70}")
            print(f"Model: {args.model}  Task: {args.task}  Clip: {'Yes' if not args.no_clip else 'No'}\n")
            if args.suite in ("mt10", "mt50"):
                result = baseline_eval.visualize_success_fail_mt10(
                    full_model_path,
                    task_name=args.task,
                    n_each=args.visualize_success_fail,
                    clip_actions=not args.no_clip,
                    seed=args.seed if args.seed is not None else 42,
                    device=device,
                    suite=args.suite,
                )
            else:
                result = baseline_eval.visualize_success_fail(
                    full_model_path,
                    n_each=args.visualize_success_fail,
                    task_name=args.task,
                    clip_actions=not args.no_clip,
                    seed=args.seed if args.seed is not None else 42,
                    device=device,
                )
            if result is not None:
                print(f"\nOverall success rate: {result:.2f}%")
                _log_test_to_wandb(
                    use_wandb=not args.no_wandb,
                    approach=args.approach,
                    model=args.model,
                    task_or_suite=args.suite if args.suite in ("mt10", "mt50") else args.task,
                    episodes=50,
                    seed=args.seed,
                    clip_actions=not args.no_clip,
                    result_single=result,
                    device=device,
                    tags=args.wandb_tag,
                )
            sys.exit(0 if result is not None else 1)

        if args.visualize_parallel > 0:
            print(f"\n{'='*70}")
            print(f"Parallel visualize: {args.visualize_parallel} envs (different goals)")
            print(f"{'='*70}")
            print(f"Model: {args.model}  Clip: {'Yes' if not args.no_clip else 'No'}\n")
            result = baseline_eval.test_policy_parallel_visualize(
                full_model_path,
                n_parallel=args.visualize_parallel,
                task_name=args.task,
                clip_actions=not args.no_clip,
                device=device,
            )
            if result is not None:
                print(f"\nSUCCESS RATE (this run): {result:.2f}%")
                _log_test_to_wandb(
                    use_wandb=not args.no_wandb,
                    approach=args.approach,
                    model=args.model,
                    task_or_suite=args.task,
                    episodes=args.visualize_parallel,
                    seed=args.seed,
                    clip_actions=not args.no_clip,
                    result_single=result,
                    device=device,
                    tags=args.wandb_tag,
                )
            sys.exit(0 if result is not None else 1)

        if args.suite in ("mt10", "mt50"):
            task_list = baseline_eval.get_tasks(args.suite)
            n_tasks = len(task_list)
            print(f"\n{'='*70}")
            print(f"Testing Policy ({args.suite.upper()})")
            print(f"{'='*70}")
            print(f"Model:           {args.model}")
            print(f"Suite:           {args.suite} (50 goals x {n_tasks} tasks)")
            print(f"Clip Actions:    {'Yes' if not args.no_clip else 'No'}")
            print(f"Seed:            {args.seed}")
            print(f"Device:          {device}")
            print(f"{'='*70}\n")
            result = baseline_eval.test_policy_multitask(
                full_model_path,
                suite=args.suite,
                clip_actions=not args.no_clip,
                seed=args.seed,
                verbose=args.verbose,
                device=device,
            )
            if result is not None:
                success_per_task, avg = result
                print("\nPer-task success rate (%):")
                for name, rate in zip(task_list, success_per_task):
                    print(f"  {name}: {rate:.1f}%")
                print(f"\n{'='*70}")
                print(f"Average success rate: {avg:.2f}%")
                print(f"{'='*70}\n")
                _log_test_to_wandb(
                    use_wandb=not args.no_wandb,
                    approach=args.approach,
                    model=args.model,
                    task_or_suite=args.suite,
                    episodes=50 * n_tasks,
                    seed=args.seed,
                    clip_actions=not args.no_clip,
                    result_mt=(success_per_task, avg),
                    task_list=task_list,
                    device=device,
                    tags=args.wandb_tag,
                )
            else:
                print("\nTest failed\n")
            sys.exit(0 if result is not None else 1)

        # Plain MT1 test (with optional visualize / visualize-series)
        if args.visualize_series > 0:
            args.visualize = True
            args.episodes = args.visualize_series
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
        print(f"Clip Actions:    {'Yes' if not args.no_clip else 'No'}")
        seed_str = str(args.seed) if args.seed is not None else "None (stochastic — results will vary run-to-run)"
        print(f"Seed:            {seed_str}")
        print(f"Visualize:       {'Yes' if args.visualize else 'No'}")
        print(f"Device:          {device}")
        print(f"{'='*70}\n")
        result = baseline_eval.test_policy(
            full_model_path,
            num_episodes=args.episodes,
            task_name=args.task,
            clip_actions=not args.no_clip,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
            visualize=args.visualize,
        )
        if result is not None:
            print(f"\n{'='*70}")
            print(f"SUCCESS RATE: {result:.2f}%")
            print(f"{'='*70}\n")
            _log_test_to_wandb(
                use_wandb=not args.no_wandb,
                approach=args.approach,
                model=args.model,
                task_or_suite=args.task,
                episodes=args.episodes,
                seed=args.seed,
                clip_actions=not args.no_clip,
                result_single=result,
                device=device,
                tags=args.wandb_tag,
            )
        else:
            print(f"\nTest failed\n")
        sys.exit(0 if result is not None else 1)

    # Non-baseline approaches: use in-script test_policy, load_model_class, etc.
    if args.visualize_success_fail > 0:
        print(f"\n{'='*70}")
        print(f"Visualize N success + N fail (N={args.visualize_success_fail})")
        print(f"{'='*70}")
        print(f"Model: {args.model}  Clip: {'Yes' if not args.no_clip else 'No'}\n")
        result = visualize_success_fail(
            approach=args.approach,
            model_name=args.model,
            n_each=args.visualize_success_fail,
            task_name=args.task,
            clip_actions=not args.no_clip
        )
        if result is not None:
            print(f"\nOverall success rate: {result:.2f}%")
            _log_test_to_wandb(
                use_wandb=not args.no_wandb,
                approach=args.approach,
                model=args.model,
                task_or_suite=args.task,
                episodes=50,
                seed=args.seed,
                clip_actions=not args.no_clip,
                result_single=result,
                tags=args.wandb_tag,
            )
        sys.exit(0 if result is not None else 1)
    
    if args.visualize_series > 0:
        args.visualize = True
        args.episodes = args.visualize_series
    
    if args.visualize_parallel > 0:
        print(f"\n{'='*70}")
        print(f"Parallel visualize: {args.visualize_parallel} envs (different goals)")
        print(f"{'='*70}")
        print(f"Model: {args.model}  Clip: {'Yes' if not args.no_clip else 'No'}\n")
        result = test_policy_parallel_visualize(
            approach=args.approach,
            model_name=args.model,
            n_parallel=args.visualize_parallel,
            task_name=args.task,
            clip_actions=not args.no_clip
        )
        if result is not None:
            print(f"\nSUCCESS RATE (this run): {result:.2f}%")
            _log_test_to_wandb(
                use_wandb=not args.no_wandb,
                approach=args.approach,
                model=args.model,
                task_or_suite=args.task,
                episodes=args.visualize_parallel,
                seed=args.seed,
                clip_actions=not args.no_clip,
                result_single=result,
                tags=args.wandb_tag,
            )
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
    print(f"Clip Actions:    {'Yes' if not args.no_clip else 'No'}")
    seed_str = str(args.seed) if args.seed is not None else "None (stochastic — results will vary run-to-run)"
    print(f"Seed:            {seed_str}")
    print(f"Visualize:       {'Yes' if args.visualize else 'No'}")
    print(f"{'='*70}\n")
    
    result = test_policy(
        approach=args.approach,
        model_name=args.model,
        num_episodes=args.episodes,
        task_name=args.task,
        clip_actions=not args.no_clip,
        verbose=args.verbose,
        visualize=args.visualize,
        eval_seed=args.seed
    )
    
    if result is not None:
        print(f"\n{'='*70}")
        print(f"SUCCESS RATE: {result:.2f}%")
        print(f"{'='*70}\n")
        _log_test_to_wandb(
            use_wandb=not args.no_wandb,
            approach=args.approach,
            model=args.model,
            task_or_suite=args.task,
            episodes=args.episodes,
            seed=args.seed,
            clip_actions=not args.no_clip,
            result_single=result,
            tags=args.wandb_tag,
        )
    else:
        print(f"\nTest failed\n")
