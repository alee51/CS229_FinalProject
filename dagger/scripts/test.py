"""
DAgger test stub. Uses core for get_tasks and env when implemented.
Root test.py imports ClonePolicy from train; for full eval use core.run_50_goal_eval.
"""
import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from core.tasks import get_tasks

__all__ = ["get_tasks"]
