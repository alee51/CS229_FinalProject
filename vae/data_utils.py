import importlib
import numpy as np
import metaworld

# Special-case mismatches between task names and policy class names in your metaworld version
SPECIAL_TASK_TO_POLICY = {
    "peg-insert-side-v3": "SawyerPegInsertionSideV3Policy",
}

def make_benchmark(benchmark: str):
    benchmark = benchmark.upper()
    if benchmark == "MT10":
        bench = metaworld.MT10()
        task_names = list(bench.train_classes.keys())
        return bench, task_names
    raise ValueError(f"Unknown benchmark '{benchmark}'. Supported: MT10.")

def _to_camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))

def _candidate_policy_class_names(task_name: str):
    if task_name in SPECIAL_TASK_TO_POLICY:
        return [SPECIAL_TASK_TO_POLICY[task_name]]

    base = task_name.replace("-v3", "").replace("-", "_")
    camel = _to_camel(base)
    return [
        f"Sawyer{camel}V3Policy",
        f"Sawyer{camel}Policy",
    ]

def get_expert_policy(task_name: str):
    policies_mod = importlib.import_module("metaworld.policies")
    for cls_name in _candidate_policy_class_names(task_name):
        if hasattr(policies_mod, cls_name):
            return getattr(policies_mod, cls_name)()
    raise ValueError(
        f"Could not find an expert policy for task '{task_name}'. "
        f"Tried: {', '.join(_candidate_policy_class_names(task_name))}."
    )

def collect_expert_data_mt(
    benchmark: str = "MT10",
    num_episodes_per_task: int = 100,
    use_all_train_task_variations: bool = True,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    bench, task_names = make_benchmark(benchmark)

    all_obs, all_actions, all_task_ids = [], [], []
    task_names_used = []

    for task_name in task_names:
        expert = get_expert_policy(task_name)

        env_cls = bench.train_classes[task_name]
        env = env_cls()

        matching_tasks = [t for t in bench.train_tasks if t.env_name == task_name]
        if not matching_tasks:
            continue

        task_id = len(task_names_used)
        task_names_used.append(task_name)

        for ep in range(num_episodes_per_task):
            task = (
                matching_tasks[ep % len(matching_tasks)]
                if use_all_train_task_variations
                else matching_tasks[int(rng.integers(0, len(matching_tasks)))]
            )
            env.set_task(task)

            obs, _ = env.reset()
            done = False
            while not done:
                action = np.clip(expert.get_action(obs), -1.0, 1.0)

                all_obs.append(obs)
                all_actions.append(action)
                all_task_ids.append(task_id)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

    return (
        np.asarray(all_obs, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        np.asarray(all_task_ids, dtype=np.int64),
        task_names_used,
    )