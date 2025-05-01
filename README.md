# 🏁 F1 Race Winner Prediction

This project uses historical Formula 1 race data to predict whether a driver will win a race. We apply a custom implementation of logistic regression trained with gradient descent, focusing on driver performance metrics and race attributes.

---

## 📊 Features Used
The following features were engineered from the dataset:

- 🛑 Number of pit stops (`pit_count`)
- 🏆 Points scored (`points`)
- 🔁 Laps completed (`laps`)
- ⏱️ Duration of pit stops in milliseconds (`duration_ms`)
- 👴 Driver's age at race time (`age`)

---

## 🚀 How to Run

Clone the repository and run the training script:

```bash
git clone https://github.com/YOUR_USERNAME/f1-race-winner-prediction.git
cd f1-race-winner-prediction/src
python train.py
