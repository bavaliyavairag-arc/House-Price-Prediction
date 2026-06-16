# ================================================
# Task-01: House Price Prediction
# Linear Regression Model
# Prodigy InfoTech ML Internship
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 1. Load Dataset ──────────────────────────────
# NOTE: Put train.csv in the same folder as this file
# Download train.csv from:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

df = pd.read_csv('train.csv')
print("✅ Dataset loaded!")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ── 2. Select Features ───────────────────────────
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target   = 'SalePrice'

df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

print(f"\n✅ Features selected: {features}")
print(f"   Target: {target}")
print(f"   Total samples after cleaning: {len(df_model)}")

# ── 3. Split into Train & Test ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✅ Data split done!")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# ── 4. Scale Features ────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train Model ───────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\n✅ Model trained successfully!")

# ── 6. Evaluate Model ────────────────────────────
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n========================================")
print("          MODEL RESULTS")
print("========================================")
print(f"  R² Score : {r2:.4f}  ({r2*100:.1f}% accuracy)")
print(f"  RMSE     : ${rmse:,.0f}")
print("========================================")

coef_df = pd.DataFrame({
    'Feature'    : features,
    'Coefficient': model.coef_
})
print("\nFeature Importance:")
print(coef_df.to_string(index=False))
print(f"Intercept: ${model.intercept_:,.0f}")

# ── 7. Plot 1: Actual vs Predicted ───────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', edgecolors='white', s=60)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)', fontsize=12)
plt.ylabel('Predicted Price ($)', fontsize=12)
plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()
print("\n✅ Plot saved: actual_vs_predicted.png")

# ── 8. Plot 2: Feature vs Price ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = ['steelblue', 'seagreen', 'tomato']
xlabels = ['Square Footage (sq ft)', 'Bedrooms', 'Bathrooms']

for i, (feat, color, xlabel) in enumerate(zip(features, colors, xlabels)):
    axes[i].scatter(X[feat], y, alpha=0.4, color=color, s=20)
    axes[i].set_xlabel(xlabel, fontsize=11)
    axes[i].set_ylabel('Sale Price ($)', fontsize=11)
    axes[i].set_title(f'{xlabel} vs Price', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_plots.png', dpi=150)
plt.show()
print("✅ Plot saved: feature_plots.png")

# ── 9. Predict a Sample House ────────────────────
print("\n========================================")
print("        SAMPLE PREDICTION")
print("========================================")
sample = pd.DataFrame({
    'GrLivArea'    : [2000],
    'BedroomAbvGr' : [3],
    'FullBath'     : [2]
})
sample_scaled     = scaler.transform(sample)
predicted_price   = model.predict(sample_scaled)[0]
print(f"  House  : 2000 sqft | 3 beds | 2 baths")
print(f"  Predicted Price: ${predicted_price:,.0f}")
print("========================================")
print("\n✅ All done! Check your folder for the saved plots.")