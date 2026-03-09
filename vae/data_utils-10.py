import numpy as np
import metaworld

# Import scripted expert policies (extend as needed for more tasks)
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
from metaworld.policies.sawyer_push_v3_policy import SawyerPushV3Policy
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
from metaworld.policies.sawyer_drawer_close_v3_policy import SawyerDrawerCloseV3Policy
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy
from metaworld.policies.sawyer_peg_insert_side_v3_policy import SawyerPegInsertSideV3Policy
from metaworld.policies.sawyer_window_open_v3_policy import SawyerWindowOpenV3Policy
from metaworld.policies.sawyer_window_close_v3_policy import SawyerWindowCloseV3Policy


# Map task name -> expert policy constructor
EXPERT_POLICY_REGISTRY = {
    "reach-v3": SawyerReachV3Policy,
    "push-v3": SawyerPushV3Policy,
    "pick-place-v3": SawyerPickPlaceV3Policy,
    "door-open-v3": SawyerDoorOpenV3Policy,
    "drawer-open-v3": SawyerDrawerOpenV3Policy,
    "drawer-close-v3": SawyerDrawerCloseV3Policy,
    "button-press-v3": SawyerButtonPressV3Policy,
    "peg-insert-side-v3": SawyerPegInsertSideV3Policy,
    "window-open-v3": SawyerWindowOpenV3Policy,
    "window-close-v3": SawyerWindowCloseV3Policy,
}


def make_benchmark(benchmark: str):
    """
    Returns a MetaWorld benchmark object and a list of task names for that benchmark.
    """
    benchmark = benchmark.upper()
    if benchmark == "MT1":
        raise ValueError("MT1 requires a specific task name. Use make_mt1(task_name) instead.")
    if benchmark == "MT10":
        bench = metaworld.MT10()
        task_names = list(bench.train_classes.keys())
        return bench, task_names
    raise ValueError(f"Unknown benchmark '{benchmark}'. Supported: MT10.")


def make_mt1(task_name: str):
    mt1 = metaworld.MT1(task_name)
    return mt1, [task_name]


def get_expert_policy(task_name: str):
    if task_name not in EXPERT_POLICY_REGISTRY:
        raise ValueError(
            f"No expert policy registered for task '{task_name}'. "
            f"Add it to EXPERT_POLICY_REGISTRY in vae/data_utils.py."
        )
    return EXPERT_POLICY_REGISTRY[task_name]()


def collect_expert_data_mt(
    benchmark: str = "MT10",
    num_episodes_per_task: int = 100,
    use_all_train_task_variations: bool = True,
    seed: int = 0,
):
    """
    Collect expert demonstrations for a multi-task benchmark (MT10).

    Returns:
      obs: (N, 39)
      acts: (N, 4)
      task_ids: (N,) integer task index (optional for conditioning later)
      task_name_list: list of task names in the benchmark ordering
    """
    rng = np.random.default_rng(seed)

    bench, task_names = make_benchmark(benchmark)

    all_obs = []
    all_actions = []
    all_task_ids = []

    for task_id, task_name in enumerate(task_names):
        env_cls = bench.train_classes[task_name]
        env = env_cls()
        expert = get_expert_policy(task_name)

        tasks = bench.train_tasks
        # Filter to tasks matching this task_name
        matching_tasks = [t for t in tasks if t.env_name == task_name]
        if len(matching_tasks) == 0:
            raise RuntimeError(f"No train_tasks found for env_name={task_name}")

        for ep in range(num_episodes_per_task):
            if use_all_train_task_variations:
                task = matching_tasks[ep % len(matching_tasks)]
            else:
                task = matching_tasks[int(rng.integers(0, len(matching_tasks)))]

            env.set_task(task)
            obs, _ = env.reset()
            done = False

            while not done:
                action = expert.get_action(obs)
                action = np.clip(action, -1.0, 1.0)

                all_obs.append(obs)
                all_actions.append(action)
                all_task_ids.append(task_id)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

    return (
        np.asarray(all_obs, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        np.asarray(all_task_ids, dtype=np.int64),
        task_names,
    )