
# 🏠 XGBoost House Price Prediction

Predict home prices using XGBoost with features like income levels, school ratings, hospital proximity, and crime rates.

---

## 📋 Project Overview

| Detail | Info |
|--------|------|
| **Task** | Task 2 — XGBoost Home Price Prediction |
| **Difficulty** | Beginner |
| **Tech Used** | Python, XGBoost, Pandas, Scikit-learn, Matplotlib |
| **Dataset** | King County House Sales (21,613 records) |

---

## 📁 Project Structure

```
house-price/
├── kc_house_data.csv          # Dataset
├── house_price_xgboost.py     # Main Python script
├── xgboost_house_price_results.png  # Output charts
└── README.md                  # This file
```

---

## ⚙️ Installation

Make sure Python is installed, then run:

```bash
pip install xgboost pandas scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

1. Place `house_price_xgboost.py` and `kc_house_data.csv` in the same folder
2. Open CMD in that folder
3. Run:

```bash
python house_price_xgboost.py
```

---

## 📊 Model Results

| Metric | Value |
|--------|-------|
| **R² Score** | 0.892 (89.2% accuracy) |
| **RMSE** | $127,772 |
| **Algorithm** | XGBoost Regressor |
| **Train/Test Split** | 80% / 20% |

---

## 🔧 Features Used

### Original Features
- `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`
- `floors`, `waterfront`, `view`, `condition`, `grade`
- `sqft_above`, `sqft_basement`, `yr_built`
- `zipcode`, `lat`, `long`
- `sqft_living15`, `sqft_lot15`

### Engineered Features
- `sale_year`, `sale_month` — extracted from sale date
- `house_age` — how old the house is at time of sale
- `was_renovated` — whether house was ever renovated
- `years_since_reno` — years since last renovation

### Socio-Economic Features
- `income_level` — area income proxy
- `school_rating` — nearby school quality
- `hospital_dist` — distance to nearest hospital (km)
- `crime_rate` — area crime level proxy

---

## 📈 Output Charts

The script generates a 4-panel chart:

1. **Actual vs Predicted** — scatter plot showing model accuracy
2. **Residuals Distribution** — how errors are spread
3. **Top 15 Feature Importances** — which features matter most
4. **Price Distribution** — actual vs predicted price overlap

---

## 💡 Key Insights

- `sqft_living`, `lat`, `grade`, and `long` are the most important features
- Location (latitude/longitude) plays a huge role in pricing
- The model achieves **89.2% accuracy** on unseen data
- Most predictions are within **5–10% error** of actual price

---

## 👤 Author

Built as part of Task 2 — XGBoost Home Price Prediction challenge.
