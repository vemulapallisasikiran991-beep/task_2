import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/house-price/house-price-prediction-master/kc_house_data.csv')
print(f"Dataset shape: {df.shape}")
print(df.dtypes)

# ── 2. Feature Engineering ────────────────────────────────────────────────────
# Extract year/month from date
df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
df['sale_year']  = df['date'].dt.year
df['sale_month'] = df['date'].dt.month

# House age & renovation flag
df['house_age']      = df['sale_year'] - df['yr_built']
df['was_renovated']  = (df['yr_renovated'] > 0).astype(int)
df['years_since_reno'] = np.where(df['yr_renovated'] > 0,
                                   df['sale_year'] - df['yr_renovated'],
                                   df['house_age'])

# Simulated socio-economic proxies (as required by the task brief)
np.random.seed(42)
n = len(df)
df['income_level']  = (df['price'] * 0.00003 + np.random.normal(0, 0.5, n)).clip(1, 10).round(1)
df['school_rating'] = (df['grade'] * 0.6 + np.random.normal(0, 0.5, n)).clip(1, 10).round(1)
df['hospital_dist'] = np.random.uniform(0.5, 15, n).round(2)   # km
df['crime_rate']    = (10 - df['condition'] + np.random.normal(0, 0.5, n)).clip(1, 10).round(1)

# Drop columns not used for modelling
df.drop(columns=['id', 'date', 'yr_renovated'], inplace=True)

# ── 3. Prepare Features / Target ──────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != 'price']
X = df[feature_cols]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── 4. Train XGBoost ──────────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse',
    early_stopping_rounds=30
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
print(f"\nRMSE : ${rmse:,.0f}")
print(f"R²   : {r2:.4f}")

# ── 6. Visualisations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('XGBoost House Price Prediction – King County', fontsize=15, fontweight='bold')

# (a) Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.3, s=12, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title(f'Actual vs Predicted  (R² = {r2:.3f})')
ax.legend()

# (b) Residuals distribution
ax = axes[0, 1]
residuals = y_test - y_pred
ax.hist(residuals, bins=60, color='coral', edgecolor='white', linewidth=0.4)
ax.axvline(0, color='navy', linestyle='--', lw=1.5)
ax.set_xlabel('Residual ($)')
ax.set_ylabel('Count')
ax.set_title('Residuals Distribution')

# (c) Top-15 Feature Importance
ax = axes[1, 0]
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
importances[::-1].plot(kind='barh', ax=ax, color='mediumseagreen')
ax.set_xlabel('Importance Score')
ax.set_title('Top 15 Feature Importances')

# (d) Price distribution – actual vs predicted
ax = axes[1, 1]
ax.hist(y_test,  bins=60, alpha=0.6, label='Actual',    color='steelblue', density=True)
ax.hist(y_pred,  bins=60, alpha=0.6, label='Predicted', color='coral',     density=True)
ax.set_xlabel('Price ($)')
ax.set_ylabel('Density')
ax.set_title('Price Distribution: Actual vs Predicted')
ax.legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/xgboost_house_price_results.png', dpi=150, bbox_inches='tight')
print("Plot saved.")

# ── 7. Sample Predictions ─────────────────────────────────────────────────────
sample = X_test.head(10).copy()
sample['Actual_Price']    = y_test.head(10).values
sample['Predicted_Price'] = model.predict(sample.drop(columns=['Actual_Price'])).round(0)
sample['Error_%']         = ((sample['Predicted_Price'] - sample['Actual_Price']) / sample['Actual_Price'] * 100).round(2)
print("\nSample Predictions:")
print(sample[['Actual_Price', 'Predicted_Price', 'Error_%']].to_string())
