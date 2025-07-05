# 🚗 EV Registrations Machine Learning Project (UK)

This project analyzes and predicts electric vehicle (EV) registrations in the UK using various machine learning models like Linear Regression, Random Forest, and XGBoost.

---

## 📊 Dataset Overview

- **Total Entries**: 4188 rows
- **Original Features**:
  - `monthOfFirstRegistration` (Date)
  - `make` (Car Brand)
  - `vehicleType` (Type of vehicle)
  - `fuelType` (Electric, Hybrid, etc.)
  - `registrations` (Number of EVs registered)

---

## 🧠 Feature Engineering

- Extracted parts from date: `year`, `month`, `quarter`, `day_of_week`
- Cumulative registrations by brand: `cumulative_registrations`
- Total brand registrations: `total_by_brand`
- Market share by brand/year: `market_share`
- 3-month rolling average: `rolling_avg_3`
- Converted categorical values (`make`, `vehicleType`, `fuelType`) using:
  - ✅ Label Encoding for Tree models
  - ✅ One-Hot Encoding for Linear models

---

## 🧪 Models Used

| Model             | MAE   | RMSE  | R² Score |
|------------------|-------|-------|----------|
| Linear Regression| 382.91| 595.41| 0.2233   |
| Random Forest    | 223.90| 501.65| 0.4486   |
| XGBoost          | 205.40| 436.24| 0.5831 ✅ |

✅ **Best Model**: XGBoost (58.31% accuracy)

---

## 📈 Visualizations

- Bar charts comparing MAE, RMSE, and R² for all models
- Time-based trends of EV registrations

---

## 📂 Project Structure

