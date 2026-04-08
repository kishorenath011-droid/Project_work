from env.environment import InboxEnv
from env.models import Email
import random

# Load some sample emails for testing
emails = [
    Email(id="1", subject="Test", body="This is a test email", sender="alice@example.com"),
    Email(id="2", subject="Hello", body="Another test", sender="bob@example.com")
]

env = InboxEnv(emails)
obs = env.reset()
done = False

# Run a loop for a few steps
while not done:
    action = random.randint(0, 3)  # Random action between 0 and 3
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

print("Test completed.")