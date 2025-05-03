import praw
import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

samples = []
for submission in reddit.subreddit("CasualConversation").top(limit=100):
    if submission.selftext and len(submission.selftext.split()) > 30:
        samples.append({
            "prompt": "Write a casual story or opinion.",
            "model": "human",
            "temperature": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": submission.selftext
        })

with open("../data/raw/human/casual/samples.json", "w") as f:
    json.dump(samples, f, indent=2)
