import json
from env.models import Email
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass

# --- 1. Support Models (Completing the 'env.models' parts) ---

@dataclass
class Email:
    id: int
    subject: str
    body: str
    sender: str
    target_action: str  # The "correct" answer for the grader

@dataclass
class Action:
    type: str
    content: Any = None

@dataclass
class Task:
    email_id: int
    expected_action: str

class Grader:
    @staticmethod
    def evaluate(action_obj: Action, expected_task: Task) -> float:
        """Returns 1.0 if the action matches the task, else -0.1 penalty."""
        if action_obj.type == expected_task.expected_action:
            return 1.0
        return -0.1

# --- 2. The Completed Environment ---

class InboxEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, emails: List[Email]):
        super().__init__()
        self.emails = emails
        self.current_idx = 0

        # Actions: 0: classify, 1: summarize, 2: reply, 3: delete
        self.action_space = spaces.Discrete(4) 
        
        # Observation space shapes (Fixed-size byte encoding)
        self.shapes = {"subject": 100, "body": 300, "sender": 50}
        
        self.observation_space = spaces.Dict({
            "subject": spaces.Box(low=0, high=255, shape=(self.shapes["subject"],), dtype=np.uint8),
            "body": spaces.Box(low=0, high=255, shape=(self.shapes["body"],), dtype=np.uint8),
            "sender": spaces.Box(low=0, high=255, shape=(self.shapes["sender"],), dtype=np.uint8)
        })

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        self.current_idx = 0
        observation = self._get_observation()
        info = {"status": "reset"}
        return observation, info

    def _encode_string(self, text: str, max_length: int) -> np.ndarray:
        """Encodes string to fixed-size uint8 array with zero-padding."""
        encoded = text.encode('utf-8')[:max_length]
        arr = np.zeros(max_length, dtype=np.uint8)
        arr[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        return arr

    def _get_observation(self) -> Dict[str, Any]:
        """Encodes the current email (or the last one if done) for the agent."""
        idx = min(self.current_idx, len(self.emails) - 1)
        email = self.emails[idx]
        
        return {
            "subject": self._encode_string(email.subject, self.shapes["subject"]),
            "body": self._encode_string(email.body, self.shapes["body"]),
            "sender": self._encode_string(email.sender, self.shapes["sender"])
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        if self.current_idx >= len(self.emails):
             # Safety check if step is called after termination
             return self._get_observation(), 0.0, True, False, {}

        email = self.emails[self.current_idx]

        # Logic using the Models
        expected_task = Task(email_id=email.id, expected_action=email.target_action)
        action_obj = Action(type=self._action_from_int(action))
        
        # Evaluate reward via the Grader
        reward = Grader.evaluate(action_obj, expected_task)

        # Increment index
        self.current_idx += 1
        
        terminated = self.current_idx >= len(self.emails)
        truncated = False
        observation = self._get_observation()
        
        info = {
            "email_id": email.id, 
            "action_taken": action_obj.type,
            "correct": reward > 0
        }

        return observation, reward, terminated, truncated, info

    def _action_from_int(self, action: int) -> str:
        mapping = {0: "classify", 1: "summarize", 2: "reply", 3: "delete"}
        return mapping.get(action, "unknown")

    def render(self):
        idx = min(self.current_idx, len(self.emails) - 1)
        email = self.emails[idx]
        print(f"[Step {self.current_idx}] Processing ID {email.id}: {email.subject[:30]}...")

# --- 3. Example Usage ---

if __name__ == "__main__":
    # Create sample emails
    data = [
        Email(101, "Urgent Meeting", "Can we meet at 5?", "boss@corp.com", "reply"),
        Email(102, "Weekly Newsletter", "Here is your news.", "news@info.com", "summarize"),
        Email(103, "Buy Shoes!", "Cheap shoes for sale.", "spam@shop.com", "delete")
    ]

    env = InboxEnv(data)
    obs, info = env.reset()
    
    for _ in range(len(data)):
        action = env.action_space.sample()  # Randomly pick an action
        obs, reward, term, trunc, info = env.step(action)
        print(f"Action: {info['action_taken']} | Reward: {reward}")
        if term:
            print("Finished processing all emails.")
