import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading dataset...")

df = pd.read_csv(
    "dataset.csv",
    header=None
)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print(f"\nAccuracy: {acc * 100:.2f}%")

joblib.dump(
    model,
    "gesture_model.pkl"
)

print("\nModel saved -> gesture_model.pkl")