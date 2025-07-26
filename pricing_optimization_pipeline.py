# pricing_optimization_pipeline.py
# Author: Maruf Ajimati
# Final Project – BAN6800 Business Analytics Capstone
# Pricing Optimization using demand elasticity, competitor prices & preferences

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter crashes in Thonny
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import joblib

# Try XGBoost if available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# --------------------------
# CONFIG
# --------------------------
ROOT_DIR = r"C:\Users\Maruf Ajimati\Documents\Nexford Assignments\BAN6800 Business Analytics Capstone\Final Project"
INPUT_FILE = os.path.join(ROOT_DIR, "retail_price.csv")
OUTPUT_DIR = ROOT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# 1) LOAD DATA
# --------------------------
df = pd.read_csv(INPUT_FILE)
print("✅ Data loaded:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# --------------------------
# 2) ROBUST CLEANING
# --------------------------
before_dupes = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Removed {before_dupes - df.shape[0]} duplicate rows.")

if "month_year" in df.columns:
    df["month_year"] = pd.to_datetime(df["month_year"], errors="coerce")
    df = df.sort_values("month_year")

numeric_candidates = [
    "unit_price", "qty", "volume", "total_price", "freight_price",
    "comp_1", "comp_2", "comp_3", "lag_price", "product_photos_qty",
    "product_name_lenght", "product_description_lenght", "holiday", "s"
]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

target_col = "qty" if "qty" in df.columns else "volume"

before_na = df.shape[0]
df.dropna(subset=["unit_price", target_col], inplace=True)
print(f"Dropped {before_na - df.shape[0]} rows with missing unit_price/{target_col}.")

for col in ["unit_price", target_col, "comp_1", "comp_2", "comp_3"]:
    if col in df.columns:
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = np.clip(df[col], low, high)

cleaned_csv_path = os.path.join(OUTPUT_DIR, "cleaned_retail_price.csv")
df.to_csv(cleaned_csv_path, index=False)
print(f"🧼 Cleaned dataset saved to: {cleaned_csv_path}")

# --------------------------
# 3) FEATURE PREPARATION
# --------------------------
for c in ["comp_1", "comp_2", "comp_3"]:
    if c in df.columns:
        df[f"delta_{c}"] = df[c] - df["unit_price"]

df["qty_pos"] = df[target_col].clip(lower=1e-6)
df["price_pos"] = df["unit_price"].clip(lower=1e-6)
df["log_qty"] = np.log(df["qty_pos"])
df["log_price"] = np.log(df["price_pos"])

if "month" not in df.columns and "month_year" in df.columns:
    df["month"] = df["month_year"].dt.month
if "year" not in df.columns and "month_year" in df.columns:
    df["year"] = df["month_year"].dt.year

# --------------------------
# 4) PRICE ELASTICITY
# --------------------------
def estimate_elasticity(group):
    if group["log_price"].nunique() < 2 or group["log_qty"].nunique() < 2:
        return np.nan
    X = group["log_price"].values.reshape(-1, 1)
    y = group["log_qty"].values
    lr = LinearRegression().fit(X, y)
    return lr.coef_[0]

if "product_id" in df.columns:
    elasticity = df.groupby("product_id").apply(estimate_elasticity).reset_index()
    elasticity.columns = ["product_id", "price_elasticity"]
    df = df.merge(elasticity, on="product_id", how="left")
else:
    try:
        df["price_elasticity"] = LinearRegression().fit(df[["log_price"]], df["log_qty"]).coef_[0]
    except:
        df["price_elasticity"] = np.nan

if "price_elasticity" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["price_elasticity"].dropna(), bins=20, kde=True)
    plt.title("Distribution of Price Elasticities")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "elasticity_distribution.png"), dpi=300)
    plt.close()

# --------------------------
# 5) FEATURES
# --------------------------
candidate_features = [
    "unit_price", "freight_price", "product_photos_qty",
    "product_name_lenght", "product_description_lenght",
    "month", "year", "holiday", "s", "lag_price",
    "price_elasticity"
]
candidate_features += [c for c in df.columns if c.startswith("delta_")]
candidate_features = [c for c in candidate_features if c in df.columns]

model_df = df.dropna(subset=candidate_features + [target_col]).copy()
X = model_df[candidate_features]
y = model_df[target_col]

# --------------------------
# 6) TRAIN / TEST SPLIT
# --------------------------
if "month_year" in model_df.columns:
    dates = model_df["month_year"].sort_values().unique()
    cutoff = int(len(dates) * 0.8)
    cutoff_date = dates[cutoff]
    train_idx = model_df["month_year"] <= cutoff_date
    test_idx = model_df["month_year"] > cutoff_date
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# --------------------------
# 7) MODELS & TUNING
# --------------------------
def eval_regression(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []

# Linear Regression
lin_pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())])
lin_pipe.fit(X_train, y_train)
results.append(eval_regression("Linear Regression", y_test, lin_pipe.predict(X_test)))

# ElasticNet
enet_pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("enet", ElasticNet(random_state=42))])
param_enet = {"enet__alpha": [0.01, 0.1, 1.0], "enet__l1_ratio": [0.1, 0.5, 0.9]}
enet_gs = GridSearchCV(enet_pipe, param_grid=param_enet, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
enet_gs.fit(X_train, y_train)
results.append(eval_regression("ElasticNet (CV)", y_test, enet_gs.predict(X_test)))

# Random Forest
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(eval_regression("Random Forest", y_test, rf.predict(X_test)))

# XGBoost
if HAS_XGB:
    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    results.append(eval_regression("XGBoost", y_test, xgb.predict(X_test)))

# --------------------------
# 8) SAVE RESULTS & VISUALS
# --------------------------
res_df = pd.DataFrame(results).sort_values("RMSE")
res_path = os.path.join(OUTPUT_DIR, "model_performance.csv")
res_df.to_csv(res_path, index=False)
print("\n📊 Model Performance Summary:\n", res_df)
print(f"Saved model performance table to: {res_path}")

fig, ax = plt.subplots(figsize=(6, 1 + 0.3 * len(res_df)))
ax.axis("off")
tbl = ax.table(cellText=res_df.round(4).values,
               colLabels=res_df.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_performance_table.png"), dpi=300)
plt.close()

# Feature importance
imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=imp.head(10).values, y=imp.head(10).index, palette="viridis")
plt.title("Top 10 Features (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance.png"), dpi=300)
plt.close()

# --------------------------
# 9) SIMULATE OPTIMAL PRICE (FIXED)
# --------------------------
def simulate_optimal_price(model, product_df, feature_cols, price_col="unit_price", n_points=50):
    if len(product_df) == 0:
        return None, None, None
    p_min, p_max = product_df[price_col].quantile([0.05, 0.95])
    grid_prices = np.linspace(p_min, p_max, n_points)
    latest = product_df.iloc[[-1]][feature_cols].copy()

    revenues, preds = [], []
    for p in grid_prices:
        temp = latest.copy()
        temp[price_col] = p
        for c in ["comp_1", "comp_2", "comp_3"]:
            if f"delta_{c}" in temp.columns:
                temp[f"delta_{c}"] = temp.get(c, temp[f"delta_{c}"].iloc[0]) - p
        yhat = model.predict(temp)[0]
        revenues.append(p * max(yhat, 0))
        preds.append(yhat)

    best_idx = int(np.argmax(revenues))
    return grid_prices[best_idx], revenues[best_idx], preds[best_idx]

opt_rows = []
if "product_id" in model_df.columns:
    for pid, group in model_df.groupby("product_id"):
        bp, br, bq = simulate_optimal_price(rf, group, X_train.columns)
        if bp is not None:
            opt_rows.append([pid, bp, bq, br])

if opt_rows:
    opt_df = pd.DataFrame(opt_rows, columns=["product_id", "optimal_price", "predicted_qty", "expected_revenue"])
    opt_path = os.path.join(OUTPUT_DIR, "optimal_prices_by_product.csv")
    opt_df.to_csv(opt_path, index=False)
    print(f"💰 Optimal prices saved to: {opt_path}")

# --------------------------
# 10) SAVE BEST MODEL
# --------------------------
best_model_name = res_df.iloc[0]["model"]
best_model = rf if best_model_name == "Random Forest" else enet_gs if "ElasticNet" in best_model_name else lin_pipe
model_out = os.path.join(OUTPUT_DIR, "best_pricing_model.pkl")
joblib.dump(best_model, model_out)
print(f"✅ Saved deployment-ready model to: {model_out}")

cols_out = os.path.join(OUTPUT_DIR, "model_feature_columns.txt")
with open(cols_out, "w") as f:
    for c in X_train.columns:
        f.write(c + "\n")
print(f"✅ Saved model feature columns to: {cols_out}")

print("\n🎉 ALL DONE.")
print("Artifacts saved in:", OUTPUT_DIR)
