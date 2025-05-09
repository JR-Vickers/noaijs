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
for submission in reddit.subreddit("CasualConversation").top(limit=200):
    if submission.selftext and len(submission.selftext.split()) > 30:
        samples.append({
            "prompt": "Write a casual story or opinion.",
            "model": "human",
            "temperature": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": submission.selftext
        })

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the output file
output_path = os.path.join(script_dir, "..", "data", "raw", "human", "casual", "samples.json")
# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(samples, f, indent=2)
