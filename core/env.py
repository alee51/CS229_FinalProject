"""
Env factory for Meta-World. Create envs on demand; do not store with checkpoints.
"""
from typing import Tuple, List, Any

def make_env(task_name: str, **env_kwargs):
    """
    Create a Meta-World env and its train_tasks for the given task name.
    Returns (env, train_tasks) where train_tasks is the list of task objects
    (one per goal) for set_task/reset. Caller is responsible for closing env.
    env_kwargs are passed to the env constructor (e.g. render_mode="human").
    """
    import metaworld
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]
    try:
        env = env_cls(**env_kwargs)
    except TypeError:
        env = env_cls()
    train_tasks = mt1.train_tasks
    return env, train_tasks


def get_train_tasks_for_suite(suite: str) -> List[Tuple[str, Any]]:
    """
    Return list of (task_name, train_tasks) for each task in the suite.
    Each train_tasks is the list of goal variants (for set_task/reset).
    """
    from core.tasks import get_tasks
    import metaworld
    out = []
    for task_name in get_tasks(suite):
        mt1 = metaworld.MT1(task_name)
        out.append((task_name, mt1.train_tasks))
    return out
