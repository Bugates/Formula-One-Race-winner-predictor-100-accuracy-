import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score

def custom_group_shuffle_split(groups, test_size=0.2, random_state=None):
    np.random.seed(random_state)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    shuffled = np.random.permutation(unique_groups)

    n_test = int(np.floor(test_size * n_groups))
    test_groups = shuffled[:n_test]
    train_groups = shuffled[n_test:]

    train_idx = np.where(np.isin(groups, train_groups))[0]
    test_idx = np.where(np.isin(groups, test_groups))[0]

    return train_idx, test_idx

base_path = r"C:\Users\ntiwari\.cache\kagglehub\datasets\rohanrao\formula-1-world-championship-1950-2020\versions\24"

pit_stops = pd.read_csv(os.path.join(base_path, 'pit_stops.csv'))
results = pd.read_csv(os.path.join(base_path, 'results.csv'))
drivers = pd.read_csv(os.path.join(base_path, 'drivers.csv'))
races = pd.read_csv(os.path.join(base_path, 'races.csv'))
status = pd.read_csv(os.path.join(base_path, 'status.csv'))

pit_stops = pit_stops[['raceId', 'driverId', 'stop', 'lap', 'time', 'duration', 'milliseconds']]
results = results[['resultId', 'raceId', 'driverId', 'constructorId', 'number', 'positionOrder', 'points', 'laps', 'milliseconds', 'statusId']]
drivers = drivers[['driverId', 'driverRef', 'dob']]
races = races[['raceId', 'year', 'round', 'circuitId', 'date']]
status = status[['statusId', 'status']]

df = pit_stops.merge(results, on=['raceId', 'driverId'], how='inner')
df = df.merge(drivers, on='driverId', how='left')
df = df.merge(races, on='raceId', how='left')
df = df.merge(status, on='statusId', how='left')

df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['positionOrder', 'points'], inplace=True)

df['age'] = (df['date'] - df['dob']).dt.days / 365
df['pit_count'] = df['stop'].astype(int)
df['duration_ms'] = df['milliseconds_x']
df['target'] = (df['positionOrder'] == 1).astype(int)

# Selecting all columns for features (excluding non-numeric columns or encoded categorical ones)
features = df.select_dtypes(include=[np.number]).columns.tolist()
features.remove('target')  # Remove target column from features

X = df[features].fillna(0).values
Y = df['target'].values.reshape(-1, 1)

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

train_idx, test_idx = custom_group_shuffle_split(df['raceId'].values, test_size=0.2, random_state=42)
X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(Y, Y_hat):
    eps = 1e-9  
    return -np.mean(Y * np.log(Y_hat + eps) + (1 - Y) * np.log(1 - Y_hat + eps))

def train(X, Y, lr=0.1, epochs=1000):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0

    for i in range(epochs):
        Z = np.dot(X, W) + b
        Y_hat = sigmoid(Z)
        loss = compute_loss(Y, Y_hat)

        dW = np.dot(X.T, (Y_hat - Y)) / m
        db = np.mean(Y_hat - Y)

        W -= lr * dW
        b -= lr * db

        if i % 100 == 0:
            print(f"Epoch {i} - Loss: {loss:.4f}")
    return W, b

def predict(X, W, b):
    Z = np.dot(X, W) + b
    return (sigmoid(Z) > 0.5).astype(int)

W, b = train(X_train, Y_train, lr=0.1, epochs=1000)

Y_pred = predict(X_test, W, b)
print("\nAccuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

print("\nFeature Coefficients:")
for f, w in zip(features, W.flatten()):
    print(f"{f}: {w:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["Not 1st", "1st"], yticklabels=["Not 1st", "1st"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
