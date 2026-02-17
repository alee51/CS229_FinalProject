"""
Collect exactly 1 expert trajectory per goal (50 total) for a single task, or for
all MT-10 tasks (500 trajectories total). Single-task: saves expert_data_{task}.npz
with states/actions. MT-10: saves expert_data_mt10.npz with states, actions, task_ids.
"""
import argparse
import importlib
import metaworld
import numpy as np
import os

# MT-10 task list (source of truth for multi-task collection)
MT10_TASKS = [
    'reach-v3',
    'push-v3',
    'pick-place-v3',
    'door-open-v3',
    'door-close-v3',
    'drawer-open-v3',
    'drawer-close-v3',
    'button-press-v3',
    'lever-pull-v3',
    'window-open-v3',
]

# task_name -> (module_path, policy_class_name) for lazy import
TASK_TO_POLICY = {
    'reach-v3': ('metaworld.policies.sawyer_reach_v3_policy', 'SawyerReachV3Policy'),
    'push-v3': ('metaworld.policies.sawyer_push_v3_policy', 'SawyerPushV3Policy'),
    'pick-place-v3': ('metaworld.policies.sawyer_pick_place_v3_policy', 'SawyerPickPlaceV3Policy'),
    'door-open-v3': ('metaworld.policies.sawyer_door_open_v3_policy', 'SawyerDoorOpenV3Policy'),
    'door-close-v3': ('metaworld.policies.sawyer_door_close_v3_policy', 'SawyerDoorCloseV3Policy'),
    'drawer-open-v3': ('metaworld.policies.sawyer_drawer_open_v3_policy', 'SawyerDrawerOpenV3Policy'),
    'drawer-close-v3': ('metaworld.policies.sawyer_drawer_close_v3_policy', 'SawyerDrawerCloseV3Policy'),
    'button-press-v3': ('metaworld.policies.sawyer_button_press_v3_policy', 'SawyerButtonPressV3Policy'),
    'lever-pull-v3': ('metaworld.policies.sawyer_lever_pull_v3_policy', 'SawyerLeverPullV3Policy'),
    'window-open-v3': ('metaworld.policies.sawyer_window_open_v3_policy', 'SawyerWindowOpenV3Policy'),
}


def get_expert_policy(task_name):
    """Return an expert policy instance for the given task name (MT-10 task names)."""
    if task_name not in TASK_TO_POLICY:
        raise ValueError(f"Unknown task '{task_name}'; supported: {list(TASK_TO_POLICY.keys())}")
    mod_path, cls_name = TASK_TO_POLICY[task_name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


def _flatten_obs(obs):
    """Return 1D array of shape (39,) for storage."""
    return np.asarray(obs).flatten()


def collect_one_per_goal(task_name='reach-v3', output_dir=None, output_path=None):
    """Collect one successful expert trajectory per goal (50 total) for a single task."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    policy = get_expert_policy(task_name)

    all_states = []
    all_actions = []
    failed = []

    for goal_idx, task in enumerate(mt1.train_tasks):
        env.set_task(task)
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        obs = _flatten_obs(obs)
        done = False
        steps = 0
        episode_states = []
        episode_actions = []

        while not done and steps < 500:
            action = policy.get_action(obs)
            episode_states.append(obs)
            episode_actions.append(action)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, term, trunc, info = step_out
            else:
                obs, _, done, info = step_out
                term, trunc = done, False
            done = term or trunc
            obs = _flatten_obs(obs) if not done else obs
            steps += 1
            if info.get('success', False):
                break

        if info.get('success', False):
            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))
            print(f"  Goal {goal_idx + 1:2d}/50: success ({len(episode_states)} steps)")
        else:
            failed.append(goal_idx)
            print(f"  Goal {goal_idx + 1:2d}/50: expert failed (skip)")

    if failed:
        print(f"\nWarning: expert failed on {len(failed)} goals: {failed}")
    if len(all_states) == 0:
        raise RuntimeError("No successful trajectories collected.")

    out_path = output_path or os.path.join(output_dir, f'expert_data_{task_name}.npz')
    np.savez(
        out_path,
        states=np.array(all_states, dtype=object),
        actions=np.array(all_actions, dtype=object),
    )
    print(f"\nSaved {len(all_states)} trajectories to {out_path}")
    total_samples = sum(len(s) for s in all_states)
    print(f"Total (s,a) pairs: {total_samples}")
    return out_path


def collect_mt10_one_per_goal(output_dir=None, output_path=None):
    """
    Collect one trajectory per goal for each of the 10 MT-10 tasks (at most 500 trajectories).
    Saves states, actions, task_ids; optionally goal_indices and task_names.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    all_states = []
    all_actions = []
    all_task_ids = []
    all_goal_indices = []
    all_task_names = []
    failed_list = []  # (task_name, goal_idx)

    for task_id, task_name in enumerate(MT10_TASKS):
        print(f"\n--- Task {task_id + 1}/10: {task_name} ---")
        mt1 = metaworld.MT1(task_name)
        env = mt1.train_classes[task_name]()
        policy = get_expert_policy(task_name)
        num_goals = len(mt1.train_tasks)

        for goal_idx in range(num_goals):
            task = mt1.train_tasks[goal_idx]
            env.set_task(task)
            out = env.reset()
            obs = out[0] if isinstance(out, tuple) else out
            obs = _flatten_obs(obs)
            done = False
            steps = 0
            episode_states = []
            episode_actions = []

            while not done and steps < 500:
                action = policy.get_action(obs)
                episode_states.append(obs)
                episode_actions.append(action)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, _, term, trunc, info = step_out
                else:
                    obs, _, done, info = step_out
                    term, trunc = done, False
                done = term or trunc
                obs = _flatten_obs(obs) if not done else obs
                steps += 1
                if info.get('success', False):
                    break

            if info.get('success', False):
                all_states.append(np.array(episode_states))
                all_actions.append(np.array(episode_actions))
                all_task_ids.append(task_id)
                all_goal_indices.append(goal_idx)
                all_task_names.append(task_name)
                print(f"  Goal {goal_idx + 1:2d}/{num_goals}: success ({len(episode_states)} steps)")
            else:
                failed_list.append((task_name, goal_idx))
                print(f"  Goal {goal_idx + 1:2d}/{num_goals}: expert failed (skip)")

    if failed_list:
        print(f"\nWarning: expert failed on {len(failed_list)} (task, goal) pairs (first 20): {failed_list[:20]}")

    if len(all_states) == 0:
        raise RuntimeError("No successful trajectories collected.")

    # Validation
    n = len(all_states)
    assert len(all_actions) == n and len(all_task_ids) == n
    assert all(0 <= t < 10 for t in all_task_ids)
    pairs = list(zip(all_task_ids, all_goal_indices))
    assert len(pairs) == len(set(pairs)), "Duplicate (task_id, goal_index) found"
    for i in range(n):
        assert all_states[i].ndim == 2 and all_states[i].shape[1] == 39, f"states[{i}].shape = {all_states[i].shape}"
        assert all_actions[i].ndim == 2 and all_actions[i].shape[1] == 4, f"actions[{i}].shape = {all_actions[i].shape}"

    out_path = output_path or os.path.join(output_dir, 'expert_data_mt10.npz')
    np.savez(
        out_path,
        states=np.array(all_states, dtype=object),
        actions=np.array(all_actions, dtype=object),
        task_ids=np.array(all_task_ids, dtype=np.int64),
        goal_indices=np.array(all_goal_indices, dtype=np.int64),
        task_names=np.array(all_task_names, dtype=object),
    )
    total_samples = sum(len(s) for s in all_states)
    per_task = [sum(1 for t in all_task_ids if t == k) for k in range(10)]
    print(f"\nSaved {n} trajectories to {out_path}")
    print(f"Total (s,a) pairs: {total_samples}")
    print(f"Per task: {dict(zip(MT10_TASKS, per_task))}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect one expert trajectory per goal (50 per task). Single task or MT-10.'
    )
    parser.add_argument(
        '--mt10',
        action='store_true',
        help='Collect for all 10 MT-10 tasks (one per goal per task); save combined npz with task_ids.',
    )
    parser.add_argument(
        '--task',
        type=str,
        default='reach-v3',
        help='Single-task mode: task name (default: reach-v3). Ignored if --mt10.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path. Default: output-dir/expert_data_{task}.npz or expert_data_mt10.npz.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory when --output is not set (default: baseline/data).',
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(script_dir, '..', 'data')
    output_dir = os.path.normpath(os.path.abspath(output_dir))

    if args.mt10:
        collect_mt10_one_per_goal(output_dir=output_dir, output_path=args.output)
    else:
        collect_one_per_goal(
            task_name=args.task,
            output_dir=output_dir,
            output_path=args.output,
        )
