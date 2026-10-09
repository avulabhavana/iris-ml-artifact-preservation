
import json
import sys

MINIMUM_ACCURACY = 0.85

print("IRIS ML QUALITY GATE")
print("--------------------")

try:
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    accuracy = metrics["accuracy"]

except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
    print("ERROR: Could not read model metrics.")
    print(error)
    sys.exit(1)

print("Model Accuracy:", round(accuracy, 4))
print("Minimum Required Accuracy:", MINIMUM_ACCURACY)

if accuracy >= MINIMUM_ACCURACY:
    print("QUALITY GATE PASSED")
    print("Model meets the accuracy requirement.")
    sys.exit(0)
else:
    print("QUALITY GATE FAILED")
    print("Model accuracy is below the required threshold.")
    sys.exit(1)
