import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Helper Functions
# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(Y, Y_hat):
    eps = 1e-9
    return -np.mean(Y * np.log(Y_hat + eps) + (1 - Y) * np.log(1 - Y_hat + eps))

def train(X, Y, lr=0.1, epochs=1000):
    m, n = X.shape
    W, b = np.zeros((n, 1)), 0
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
    return (sigmoid(np.dot(X, W) + b) > 0.5).astype(int)

def predict_proba(X, W, b):
    return sigmoid(np.dot(X, W) + b)

def custom_group_shuffle_split(groups, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    unique_groups = np.unique(groups)
    n_test = int(np.floor(test_size * len(unique_groups)))
    shuffled = np.random.permutation(unique_groups)
    test_groups = shuffled[:n_test]
    train_groups = shuffled[n_test:]
    train_idx = np.where(np.isin(groups, train_groups))[0]
    test_idx = np.where(np.isin(groups, test_groups))[0]
    return train_idx, test_idx

# ---------------------------
# Load Data
# ---------------------------
base_path = r"C:\Users\ntiwari\.cache\kagglehub\datasets\rohanrao\formula-1-world-championship-1950-2020\versions\24"

pit_stops = pd.read_csv(os.path.join(base_path, 'pit_stops.csv'))[['raceId', 'driverId', 'stop', 'lap', 'time', 'duration', 'milliseconds']]
results = pd.read_csv(os.path.join(base_path, 'results.csv'))[['raceId', 'driverId', 'constructorId', 'number', 'positionOrder', 'points', 'laps', 'milliseconds', 'statusId']]
drivers = pd.read_csv(os.path.join(base_path, 'drivers.csv'))[['driverId', 'driverRef', 'dob']]
races = pd.read_csv(os.path.join(base_path, 'races.csv'))[['raceId', 'year', 'round', 'circuitId', 'date']]
status = pd.read_csv(os.path.join(base_path, 'status.csv'))[['statusId', 'status']]

# ---------------------------
# Merge and Clean
# ---------------------------
df = pit_stops.merge(results, on=['raceId', 'driverId'])
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

# ---------------------------
# Prepare Features
# ---------------------------
features = ['pit_count', 'points', 'laps', 'duration_ms', 'age']
X = df[features].fillna(0).values
Y = df['target'].values.reshape(-1, 1)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / (X_std + 1e-8)

# ---------------------------
# Train/Test Split
# ---------------------------
train_idx, test_idx = custom_group_shuffle_split(df['raceId'].values, test_size=0.2, random_state=42)
X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# ---------------------------
# Train Model
# ---------------------------
W, b = train(X_train, Y_train, lr=0.1, epochs=1000)
Y_pred = predict(X_test, W, b)

# ---------------------------
# Evaluate Model
# ---------------------------
print("\nAccuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["Not 1st", "1st"], yticklabels=["Not 1st", "1st"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ---------------------------
# Predict for 2025 Miami GP Proxy
# ---------------------------
miami_candidates = races[races['year'] >= 2022]
likely_miami_circuit = miami_candidates['circuitId'].value_counts().idxmax()
miami_races = races[(races['circuitId'] == likely_miami_circuit) & (races['year'] >= 2022)]
latest_miami = miami_races[miami_races['year'] == miami_races['year'].max()]

if latest_miami.empty:
    print("\nCould not find a recent Miami GP in the dataset.")
else:
    race_id = latest_miami.iloc[0]['raceId']
    print(f"\nUsing raceId {race_id} from year {latest_miami.iloc[0]['year']} as proxy for 2025 Miami GP.")

    miami_df = df[df['raceId'] == race_id].copy()
    miami_X = miami_df[features].fillna(0).values
    miami_X = (miami_X - X_mean) / X_std
    miami_df['win_prob'] = predict_proba(miami_X, W, b)

    # Safely ensure 'driverRef' exists
    if 'driverRef' not in miami_df.columns or miami_df['driverRef'].isnull().all():
        miami_df = miami_df.merge(drivers[['driverId', 'driverRef']].drop_duplicates(), on='driverId', how='left')
        miami_df['driverRef'] = miami_df['driverRef'].fillna(miami_df['driverId'].astype(str))
    else:
        miami_df['driverRef'] = miami_df['driverRef'].fillna(miami_df['driverId'].astype(str))

    top_driver = miami_df.sort_values('win_prob', ascending=False).iloc[0]
    print("\n--- Top Predicted 2025 Miami GP Contender ---")
    print(f"{top_driver['driverRef']} (Prob: {top_driver['win_prob']:.3f})")
