import random
from env.environment import InboxEnv
from env.models import Email

# Load your emails (for example, using the easy set)
emails = []
with open('data/emails_easy.json', 'r') as f:
    import json
    emails = [Email(**email) for email in json.load(f)]

# Initialize the environment with your emails
env = InboxEnv(emails)

# Reset the environment
obs = env.reset()
done = False

# Loop until all emails are processed
while not done:
    # Take a random action: 0: classify, 1: summarize, 2: reply, 3: delete
    action = random.randint(0, 3)
    
    # Step the environment
    obs, reward, done, info = env.step(action)
    
    # Print the action and reward