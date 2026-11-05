import json
import sys

THRESHOLD = 0.90

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

if accuracy >= THRESHOLD:
    print(f"Accuracy {accuracy:.4f} >= {THRESHOLD}. Ready to deploy!")
else:
    print(f"Accuracy {accuracy:.4f} < {THRESHOLD}. Not good enough.")
    sys.exit(1)