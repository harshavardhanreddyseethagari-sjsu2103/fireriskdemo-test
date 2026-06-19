# view_logs.py
#
# Reads request_log.jsonl (one JSON object per line) and prints summary
# stats: total requests, error rate, average latency, and how often each
# risk category was predicted.
#
# Run it any time after hitting /predict a few times:
#   python3 view_logs.py

import json

LOG_FILE = "logs/request_log.jsonl"

successes = []
validation_errors = []
server_errors = []

# Read the file line by line. Each line is a separate, complete JSON object.
with open(LOG_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue   # skip blank lines, just in case
        event = json.loads(line)   # parse the JSON text back into a dict
        if event["status"] == "success":
            successes.append(event)
        elif event["status"] == "validation_error":
            validation_errors.append(event)
        else:
            server_errors.append(event)

total = len(successes) + len(validation_errors) + len(server_errors)

print(f"Total requests logged: {total}")
print(f"  Successful:        {len(successes)}")
print(f"  Validation errors:  {len(validation_errors)}  (bad input — e.g. out of range)")
print(f"  Server errors:       {len(server_errors)}  (something broke in our code)")

if total > 0:
    error_rate = (len(validation_errors) + len(server_errors)) / total * 100
    print(f"  Overall error rate: {error_rate:.1f}%")

if successes:
    # Average latency across all successful requests
    avg_latency = sum(e["latency_ms"] for e in successes) / len(successes)
    max_latency = max(e["latency_ms"] for e in successes)
    print(f"\nLatency — avg: {avg_latency:.2f}ms, max: {max_latency:.2f}ms")

    # Count how many times each risk category was predicted.
    # This is a manual version of what a Counter/dictionary tally does.
    category_counts = {}
    for e in successes:
        pred = e["prediction"]
        category_counts[pred] = category_counts.get(pred, 0) + 1

    print("\nPrediction breakdown:")
    for category, count in category_counts.items():
        pct = count / len(successes) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")

if validation_errors:
    print("\nRecent validation errors (bad input):")
    for e in validation_errors[-5:]:
        print(f"  [{e['timestamp']}] input={e['input']}")

if server_errors:
    print("\nRecent server errors:")
    for e in server_errors[-5:]:
        print(f"  [{e['timestamp']}] {e['error']}")