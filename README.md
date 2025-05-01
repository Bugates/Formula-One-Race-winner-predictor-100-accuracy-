# 🏎️ Formula 1 Winner Classification

This project uses Formula 1 World Championship data to predict whether a driver finishes **first** in a race based on features like pit stop duration, laps, and age. A simple **logistic regression** model is implemented from scratch in NumPy, with custom train/test splitting based on race grouping and gained 100% accuracy.

---

## 📂 Dataset

The data used is from [Kaggle's Formula 1 World Championship (1950 - 2020)](https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020).

Required CSV files:
- `pit_stops.csv`
- `results.csv`
- `drivers.csv`
- `races.csv`
- `status.csv`

Ensure all files are located in the directory specified by the `base_path` variable in the code.

---

## 🧠 Features Used

- `pit_count`: Number of pit stops
- `points`: Driver's points in the race
- `laps`: Number of laps completed
- `duration_ms`: Pit stop duration (milliseconds)
- `age`: Driver's age at race time

Target: Whether the driver **finished first** in the race (`1` if yes, else `0`)

---

## 🛠️ How It Works

1. **Data Preparation**:
   - Merges multiple datasets on race and driver IDs.
   - Computes driver age and pit stop counts.
   - Normalizes features.

2. **Custom Train-Test Split**:
   - Splits based on unique `raceId`s to avoid data leakage across races.

3. **Model Training**:
   - Implements logistic regression using gradient descent.
   - Optimizes weights to classify winners vs. non-winners.

4. **Evaluation**:
   - Prints accuracy, classification report, and feature weights.
   - Displays a confusion matrix.

---

## 📊 Sample Output

```
Epoch 0 - Loss: 0.6931
...
Epoch 900 - Loss: 0.5102

Accuracy: 0.84

Feature Coefficients:
pit_count: -0.25
points:     1.78
laps:       0.32
duration_ms:-0.11
age:       -0.02
```

---

## 📦 Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🚀 Run the Code

Edit the `base_path` variable to point to the directory where the CSVs are stored:

```python
base_path = r"your\local\path\to\csvs"
```

Then run:

```bash
python train.py
```

---

## 📁 File Structure

```text
.
├── src
      └──train.py
├── README.md
└── data/
    └── code to implement kaggle data.py 
```

---

## 🧾 License

This project is open for educational and non-commercial use. Attribution is appreciated.
