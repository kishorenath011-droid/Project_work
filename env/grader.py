import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.models import Action, Task
from typing import List


class Grader:
    """
    The Grader evaluates agent actions against expected tasks and returns a score.
    """

    @staticmethod
    def evaluate(action: Action, task: Task) -> int:
        """
        Compare an agent's action to the expected task. Return 1 if correct, 0 if incorrect.
        """
        if action.type == task.expected_action:
            # If the action type is correct, check optional content (if provided)
            if action.content and task.expected_output:
                return int(action.content.strip().lower() == task.expected_output.strip().lower())
            # If no content expected, just the action type matters
            return 1
        return 0

    @staticmethod
    def evaluate_batch(actions: List[Action], tasks: List[Task]) -> float:
        """
        Evaluate a batch of actions and tasks, return average score.
        """
        assert len(actions) == len(tasks), "Actions and tasks must be same length"
        total_score = sum(Grader.evaluate(a, t) for a, t in zip(actions, tasks))
        return total_score / len(actions)
    from env.models import Action, Task