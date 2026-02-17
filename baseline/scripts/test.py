"""
Baseline test/eval script. Canonical implementation for baseline policy evaluation.
- Invoke from project root (recommended): python test.py --approach baseline --model <name> [--suite mt10]
- Or run directly: python baseline/scripts/test.py --model <name> [--suite mt1|mt10]
Exports: test_policy, test_policy_mt10, visualize_success_fail, visualize_success_fail_mt10, test_policy_parallel_visualize, get_device, MT10_TASKS.
"""
import time
import metaworld
import torch
import numpy as np
import argparse
import sys
import os
import traceback

# Add project root so we can import baseline.tasks and baseline/scripts/train
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_BASELINE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)
from train import ClonePolicy, MT10_TASKS, one_hot_task, get_device
from baseline.tasks import get_tasks, policy_input_dim

# #region agent log
_DEBUG_LOG = r"c:\Users\nancy\Desktop\CS229_FinalProject\.cursor\debug.log"
def _dbg(payload):
    try:
        import json
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": __import__("time").time() * 1000, **payload}) + "\n")
    except Exception:
        pass
# #endregion


def test_policy(model_path, num_episodes=100, task_name='reach-v3', clip_actions=True, verbose=0, seed=None, device=None,
                return_episode_results=False, visualize=False, goal_indices=None, max_steps=500, print_timing=True):
    """Test a policy model (single-task, 39-dim).

    Args:
        model_path: Path to the saved policy model (can be relative or absolute)
        num_episodes: Number of episodes to test
        task_name: MetaWorld task name
        clip_actions: Whether to clip actions to [-1, 1] (default True; match train.py)
        verbose: Print progress every N episodes (0=off)
        seed: If set, env.reset(seed=seed+goal_idx) for reproducibility (match train eval)
        device: torch device or None (uses get_device('auto'))
        return_episode_results: If True, return (success_rate, [(goal_idx, success), ...])
        visualize: If True, render env and sleep each step
        goal_indices: If set, run only these goal indices (in order); overrides num_episodes when used
        max_steps: Max steps per episode (default 500)
        print_timing: If True, print timing line at end (default True)
    Returns:
        success_rate, or (success_rate, episode_results) if return_episode_results
    """
    t_start = time.perf_counter()
    if device is None:
        device = get_device("auto")

    # Resolve model path - if relative, look in ../models/
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        model_path = os.path.join(_SCRIPT_DIR, "..", "models", model_path)

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
    except Exception as e:
        print(f"❌ Failed to load task '{task_name}': {e}")
        return None

    t_load_start = time.perf_counter()
    try:
        model = ClonePolicy(39, 4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
    except Exception as e:
        print(f"❌ Failed to load model '{model_path}': {e}")
        return None
    t_load_end = time.perf_counter()

    model.eval()
    success_count = 0
    episode_results = []
    t_rollout_start = time.perf_counter()

    n_goals = len(mt1.train_tasks)
    if goal_indices is not None:
        indices_to_run = goal_indices
    else:
        indices_to_run = [i % n_goals for i in range(num_episodes)]
    actual_episodes = len(indices_to_run)

    for i, goal_idx in enumerate(indices_to_run):
        task = mt1.train_tasks[goal_idx]
        env.set_task(task)
        if seed is not None:
            try:
                out = env.reset(seed=seed + goal_idx)
            except TypeError:
                out = env.reset()
        else:
            out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        if isinstance(obs, tuple):
            obs = obs[0] if len(obs) > 0 else obs
        obs = np.asarray(obs).flatten()
        done = False
        steps = 0

        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()
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
            steps += 1
            obs = np.asarray(obs).flatten() if not done else obs

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
            print(f"  Progress: {i+1}/{actual_episodes} | Current rate: {success_count/(i+1)*100:.1f}%")
        if visualize:
            print(f"  Episode {i+1} (goal {goal_idx}): {'success' if succ else 'fail'}")

    if visualize:
        try:
            env.close()
        except Exception:
            pass

    t_end = time.perf_counter()
    success_rate = success_count / actual_episodes * 100
    if print_timing:
        total_s = t_end - t_start
        load_s = t_load_end - t_load_start
        rollout_s = t_end - t_rollout_start
        print(f"\nTiming: total={total_s:.2f}s  model_load={load_s:.2f}s  rollout={rollout_s:.2f}s  episodes/sec={actual_episodes/rollout_s:.1f}")
    if return_episode_results:
        return success_rate, episode_results
    return success_rate


def test_policy_multitask(model_path, suite="mt10", clip_actions=True, seed=42, verbose=False, device=None):
    """Test a multi-task policy: 50 episodes (1 per goal) per task. suite in mt10, mt50.
    Returns per-task success rates and average. clip_actions defaults True to match train.py."""
    t_start = time.perf_counter()
    if device is None:
        device = get_device("auto")
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        model_path = os.path.join(_SCRIPT_DIR, "..", "models", model_path)
    in_dim = policy_input_dim(suite)
    task_list = get_tasks(suite)
    n_tasks = len(task_list)
    t_load_start = time.perf_counter()
    try:
        model = ClonePolicy(in_dim, 4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
    except Exception as e:
        print(f"Failed to load model '{model_path}': {e}")
        return None
    t_load_end = time.perf_counter()
    model.eval()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            try:
                xpu.manual_seed_all(seed)
            except Exception:
                pass
    success_per_task = []
    task_times = []
    t_rollout_start = time.perf_counter()
    for task_id, task_name in enumerate(task_list):
        t_task_start = time.perf_counter()
        try:
            mt1 = metaworld.MT1(task_name)
            env = mt1.train_classes[task_name]()
        except Exception as e:
            print(f"❌ Failed to load task '{task_name}': {e}")
            success_per_task.append(0.0)
            task_times.append(0.0)
            continue
        oh = one_hot_task(task_id, num_tasks=n_tasks)
        n_goals = min(50, len(mt1.train_tasks))
        success_count = 0
        for goal_idx in range(n_goals):
            task = mt1.train_tasks[goal_idx]
            env.set_task(task)
            if seed is not None:
                try:
                    out = env.reset(seed=seed + task_id * 1000 + goal_idx)
                except TypeError:
                    out = env.reset()
            else:
                out = env.reset()
            obs = out[0] if isinstance(out, tuple) else out
            if isinstance(obs, tuple):
                obs = obs[0] if len(obs) > 0 else obs
            obs = np.asarray(obs).flatten()
            done = False
            steps = 0
            while not done and steps < 500:
                x = np.concatenate([obs, oh]).astype(np.float32)
                obs_tensor = torch.FloatTensor(x).to(device)
                with torch.no_grad():
                    action = model(obs_tensor).cpu().numpy()
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
                obs = np.asarray(obs).flatten() if not done else obs
                steps += 1
            if info.get('success', False):
                success_count += 1
            if verbose and (goal_idx + 1) % verbose == 0:
                print(f"  {task_name} goal {goal_idx+1}/{n_goals}: {success_count/(goal_idx+1)*100:.1f}%")
        rate = success_count / n_goals * 100
        success_per_task.append(rate)
        task_times.append(time.perf_counter() - t_task_start)
        if verbose:
            print(f"  {task_name}: {rate:.1f}% ({success_count}/{n_goals})")
    t_end = time.perf_counter()
    avg = sum(success_per_task) / len(success_per_task)
    total_s = t_end - t_start
    load_s = t_load_end - t_load_start
    rollout_s = t_end - t_rollout_start
    total_episodes = n_tasks * 50
    print(f"\nTiming: total={total_s:.2f}s  model_load={load_s:.2f}s  rollout={rollout_s:.2f}s  episodes/sec={total_episodes/rollout_s:.1f}")
    print("Per-task time (s):")
    for name, sec in zip(task_list, task_times):
        print(f"  {name}: {sec:.2f}s")
    return success_per_task, avg


def test_policy_mt10(model_path, clip_actions=True, seed=42, verbose=False, device=None):
    """Backward-compat: test_policy_multitask(..., suite='mt10')."""
    return test_policy_multitask(model_path, suite="mt10", clip_actions=clip_actions, seed=seed, verbose=verbose, device=device)


def visualize_success_fail(model_path, n_each=3, task_name='reach-v3', clip_actions=True, seed=42, device=None):
    """Run 50 episodes (no render), then show n_each success + n_each fail with render. Same env so labels match.
    Returns overall success rate or None on failure."""
    print("Eval 50 goals (no render), then show {} success + {} fail in same run.\n".format(n_each, n_each))
    out = test_policy(
        model_path,
        num_episodes=50,
        task_name=task_name,
        clip_actions=clip_actions,
        verbose=0,
        seed=seed,
        device=device,
        return_episode_results=True,
        visualize=False,
        print_timing=False,
    )
    if out is None:
        return None
    rate, episode_results = out
    success_goals = [g for g, s in episode_results if s]
    fail_goals = [g for g, s in episode_results if not s]
    n_s = min(n_each, len(success_goals))
    n_f = min(n_each, len(fail_goals))
    goals_to_show = success_goals[:n_s] + fail_goals[:n_f]
    if not goals_to_show:
        print("No episodes to visualize.")
        return rate
    print(f"\nOverall (50 goals): {rate:.1f}%")
    print(f"Showing {n_s} SUCCESS (goals {goals_to_show[:n_s]}) then {n_f} FAIL (goals {goals_to_show[n_s:]}). Same env — labels match.")
    print("(Labels use the env's binary success threshold; what you see may look better or worse than the env result.)\n")
    test_policy(
        model_path,
        task_name=task_name,
        clip_actions=clip_actions,
        verbose=0,
        seed=seed,
        device=device,
        goal_indices=goals_to_show,
        num_episodes=len(goals_to_show),
        visualize=True,
        print_timing=False,
    )
    return rate


def visualize_success_fail_mt10(model_path, task_name, n_each=3, clip_actions=True, seed=42, device=None, suite="mt10"):
    """Run 50 episodes (no render) for one multi-task suite task, then show n_each success + n_each fail with render.
    Uses policy_input_dim(suite). Returns overall success rate for that task or None on failure."""
    task_list = get_tasks(suite)
    if task_name not in task_list:
        print(f"Task '{task_name}' not found for {suite}. Valid tasks: {', '.join(task_list)}.")
        return None
    print("Eval 50 goals (no render), then show {} success + {} fail in same run.\n".format(n_each, n_each))
    if device is None:
        device = get_device("auto")
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        model_path = os.path.join(_SCRIPT_DIR, "..", "models", model_path)
    in_dim = policy_input_dim(suite)
    n_tasks = len(task_list)
    try:
        model = ClonePolicy(in_dim, 4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
    except Exception as e:
        print(f"❌ Failed to load model '{model_path}': {e}")
        return None
    model.eval()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            try:
                xpu.manual_seed_all(seed)
            except Exception:
                pass
    task_id = task_list.index(task_name)
    oh = one_hot_task(task_id, num_tasks=n_tasks)
    try:
        mt1 = metaworld.MT1(task_name)
        env_cls = mt1.train_classes[task_name]
        env = env_cls()
    except Exception as e:
        print(f"❌ Failed to load task '{task_name}': {e}")
        return None
    n_goals = min(50, len(mt1.train_tasks))
    episode_results = []
    for goal_idx in range(n_goals):
        task = mt1.train_tasks[goal_idx]
        env.set_task(task)
        if seed is not None:
            try:
                out = env.reset(seed=seed + task_id * 1000 + goal_idx)
            except TypeError:
                out = env.reset()
        else:
            out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        if isinstance(obs, tuple):
            obs = obs[0] if len(obs) > 0 else obs
        obs = np.asarray(obs).flatten()
        done = False
        steps = 0
        while not done and steps < 500:
            x = np.concatenate([obs, oh]).astype(np.float32)
            obs_tensor = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()
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
            obs = np.asarray(obs).flatten() if not done else obs
            steps += 1
        episode_results.append((goal_idx, bool(info.get("success", False))))
    success_count = sum(1 for _, s in episode_results if s)
    rate = success_count / n_goals * 100
    success_goals = [g for g, s in episode_results if s]
    fail_goals = [g for g, s in episode_results if not s]
    n_s = min(n_each, len(success_goals))
    n_f = min(n_each, len(fail_goals))
    goals_to_show = success_goals[:n_s] + fail_goals[:n_f]
    if not goals_to_show:
        print("No episodes to visualize.")
        return rate
    print(f"\nOverall (50 goals for {task_name}): {rate:.1f}%")
    print(f"Showing {n_s} SUCCESS (goals {goals_to_show[:n_s]}) then {n_f} FAIL (goals {goals_to_show[n_s:]}). Same env — labels match.")
    print("(Labels use the env's binary success threshold; what you see may look better or worse than the env result.)\n")
    try:
        env_render = env_cls(render_mode="human")
    except TypeError:
        env_render = env_cls()
        print("Warning: render_mode not supported; running phase 2 without visualization.")
    for j, goal_idx in enumerate(goals_to_show):
        is_success = j < n_s
        print(f"  >>> Episode {j+1}/{len(goals_to_show)}: Goal {goal_idx} — {'SUCCESS' if is_success else 'FAIL'} <<<")
        task = mt1.train_tasks[goal_idx]
        env_render.set_task(task)
        if seed is not None:
            try:
                out = env_render.reset(seed=seed + task_id * 1000 + goal_idx)
            except TypeError:
                out = env_render.reset()
        else:
            out = env_render.reset()
        obs = out[0] if isinstance(out, tuple) else out
        if isinstance(obs, tuple):
            obs = obs[0] if len(obs) > 0 else obs
        obs = np.asarray(obs).flatten()
        done = False
        steps = 0
        while not done and steps < 500:
            x = np.concatenate([obs, oh]).astype(np.float32)
            obs_tensor = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()
            action = np.asarray(action).flatten().astype(np.float64)
            if clip_actions:
                action = np.clip(action, -1.0, 1.0)
            step_out = env_render.step(action)
            if len(step_out) == 5:
                obs, _, term, trunc, info = step_out
            else:
                obs, _, done, info = step_out
                term, trunc = done, False
            done = term or trunc
            obs = np.asarray(obs).flatten() if not done else obs
            steps += 1
            try:
                env_render.render()
            except Exception:
                pass
            time.sleep(0.02)
        print(f"  -> Env result: {'success' if info.get('success', False) else 'fail'}")
    try:
        env_render.close()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass
    return rate


def test_policy_parallel_visualize(model_path, n_parallel=5, task_name='reach-v3', clip_actions=True, device=None):
    """Run n_parallel envs at once, each with a different goal, all rendering. Steps all envs in lockstep."""
    if device is None:
        device = get_device("auto")
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '..', 'models', model_path)
    try:
        mt1 = metaworld.MT1(task_name)
        env_cls = mt1.train_classes[task_name]
        envs = [env_cls(render_mode="human") for _ in range(n_parallel)]
    except TypeError:
        print("Warning: render_mode not supported; cannot run parallel visualize.")
        return None
    except Exception as e:
        print(f"❌ Failed to load task '{task_name}': {e}")
        return None
    try:
        model = ClonePolicy(39, 4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
    except Exception as e:
        print(f"❌ Failed to load model '{model_path}': {e}")
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
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
            obs_t = torch.FloatTensor(obs_list[i]).to(device)
            with torch.no_grad():
                a = model(obs_t).cpu().numpy()
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
        for env in envs:
            try:
                env.render()
            except Exception:
                pass
        time.sleep(0.02)
        step_count += 1
    success_count = sum(1 for i in range(n_parallel) if info_list[i].get('success', False))
    print(f"\nParallel visualize ({n_parallel} goals):")
    for i in range(n_parallel):
        print(f"  Goal {i}: {'success' if info_list[i].get('success') else 'fail'}")
    print(f"  Total: {success_count}/{n_parallel}")
    for env in envs:
        try:
            env.close()
        except Exception:
            pass
    return success_count / n_parallel * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a behavioral cloning policy')
    parser.add_argument('--model', type=str, required=True, help='Path or name of the policy model (.pth file)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of test episodes (default: 100); for --suite mt10 fixed at 50 per task')
    parser.add_argument('--task', type=str, default='reach-v3', help='MetaWorld task name (default: reach-v3); ignored if --suite mt10')
    parser.add_argument("--suite", type=str, choices=["mt1", "mt10", "mt50"], default="mt1",
                        help="mt1: single task (39-dim). mt10/mt50: multi-task, 50 goals per task (policy_input_dim from suite)")
    parser.add_argument('--seed', type=int, default=None, help='Env seed for reproducibility (used with 50-goal eval)')
    parser.add_argument('--no-clip', action='store_true', help='Do not clip actions (default: clip to [-1, 1], same as train.py)')
    parser.add_argument('--verbose', type=int, default=0, help='Print progress every N episodes (0=off)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'xpu', 'cpu'],
                        help='Device: auto (prefer GPU), cuda (NVIDIA), xpu (Intel Arc), or cpu (default: auto)')
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    
    print(f"\n{'='*60}")
    print(f"Testing Policy")
    print(f"{'='*60}")
    print(f"Model:           {args.model}")
    print(f"Suite:           {args.suite}")
    print(f"Task:            {args.task}")
    print(f"Episodes:        {args.episodes}")
    print(f"Seed:            {args.seed}")
    print(f"Clip Actions:    {not args.no_clip}")
    print(f"Device:          {device}")
    print(f"{'='*60}\n")
    
    if args.suite in ("mt10", "mt50"):
        result = test_policy_multitask(
            model_path=args.model,
            suite=args.suite,
            clip_actions=not args.no_clip,
            seed=args.seed,
            verbose=args.verbose,
            device=device,
        )
        if result is not None:
            success_per_task, avg = result
            task_list = get_tasks(args.suite)
            print("\nPer-task success rate (%):")
            for name, rate in zip(task_list, success_per_task):
                print(f"  {name}: {rate:.1f}%")
            print(f"\nAverage success rate: {avg:.2f}%")
        else:
            print("\nTest failed")
    else:
        result = test_policy(
            model_path=args.model,
            num_episodes=args.episodes,
            task_name=args.task,
            clip_actions=not args.no_clip,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
        )
        if result is not None:
            print(f"\n✅ Success Rate: {result:.2f}%")
        else:
            print(f"\n❌ Test failed")
