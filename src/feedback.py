import json
from pathlib import Path

def save_feedback(image_id, feedback_data):
    Path("feedback").mkdir(exist_ok=True)
    with open(f"feedback/{image_id}.json", "w") as f:
        json.dump(feedback_data, f)
