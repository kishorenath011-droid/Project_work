from env.environment import InboxEnv

def load_emails(file_path):
    import json
    from env.models import Email
    with open(file_path, 'r') as f:
        data = json.load(f)
        return [Email(**email) for email in data]

easy_emails = load_emails('data/emails_easy.json')
medium_emails = load_emails('data/emails_medium.json')
hard_emails = load_emails('data/emails_hard.json')

__all__ = ['InboxEnv', 'easy_emails', 'medium_emails', 'hard_emails']