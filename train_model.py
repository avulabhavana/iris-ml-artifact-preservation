
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Iris dataset
df = pd.read_csv("iris.csv")

print("Dataset Shape:", df.shape)

# Separate features and target
X = df.drop(columns=["species"])
y = df["species"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build the ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200))
])

# Train the model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Model Accuracy:", round(accuracy, 4))

# Save trained model
joblib.dump(model, "iris_model.pkl")

# Save evaluation metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model saved as iris_model.pkl")
print("Metrics saved as metrics.json")
