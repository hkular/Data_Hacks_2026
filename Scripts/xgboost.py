#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 08:47:04 2026

@author: ilongoriavalenzuela
"""
# XGBoost model predicting asthma ER visits from key pollutants (from simple regression ran earlier)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import os

# Load all relevant data
try:
    # Get the directory of the current script
    script_dir = Path(os.path.abspath(__file__)).parent
    # Go up one level to the main_repo
    root_dir = script_dir.parent
except NameError:
    # Fallback: IDE working directory
    root_dir = Path(os.getcwd()).parent


air_data = pd.read_csv(f'{root_dir}/EPA_data/air_aqi_and_particles_annual/2015-2022.csv')
med_data = pd.read_csv(f'{root_dir}/Asthma_Data/cleaned_er.csv')

merge_data = pd.merge(
    air_data, med_data,
    left_on=['year','county'],
    right_on=['YEAR','COUNTY']
)

# Reshape to wide format (pollutants become columns)

subset = merge_data.loc[:, [
    'year', 'county', 'AGE',
    'AGE_ADJUSTED_ED_VISIT_RATE',
    'particle', 'avg_val'
]]

df_wide = subset.pivot_table(
    index=['year', 'county', 'AGE', 'AGE_ADJUSTED_ED_VISIT_RATE'],
    columns='particle',
    values='avg_val'
).reset_index()

print("After pivot:", df_wide.shape)

#grabbing top pollutants from regression\
# sulfur dioxide and carbon monoxide are removed from regression
selected_pollutants = [

    'Nitrogen dioxide (NO2)',
    'Nitric oxide (NO)',
    'Ozone'
]

# keep only needed columns in another copy 
df_model = df_wide[['year', 'county', 'AGE', 'AGE_ADJUSTED_ED_VISIT_RATE'] + selected_pollutants].copy()
print("Before cleaning:", df_model.shape)
print("Missing fraction per column:\n", df_model.isna().mean())


# handle missing data
df_model[selected_pollutants] = df_model[selected_pollutants].fillna(
    df_model[selected_pollutants].mean()
)

print("After filling NaNs:", df_model.shape)


# run XGBoost by age 
for age_group, group in df_model.groupby('AGE'):

    print("\n" + "="*60)
    print(f"AGE GROUP: {age_group}")
    print("Group size:", group.shape)

    X = group[selected_pollutants]
    y = group['AGE_ADJUSTED_ED_VISIT_RATE']

    print("Feature variance:\n", X.var())

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        tree_method='hist',
        random_state=42
    )

    model.fit(X_train, y_train)

    # eval model
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # plot feature importance
    plt.figure(figsize=(9, 5))
    xgb.plot_importance(model, importance_type='gain')
    plt.title(f"Feature Importance ({age_group})")

    plt.tight_layout()
    save_path = f"{root_dir}/Scripts/feature_importance_{age_group}.png"
    #plt.savefig(save_path, bbox_inches='tight')
    #plt.close()
    plt.show()


    print(f"Saved plot to: {save_path}")
    

# Part 2 - running version with counties demeaned 
# to help remove between county effects
# absolute vs. within county pollution spikes

df_demeaned = df_model.copy()

for col in selected_pollutants:
    df_demeaned[col] = (
        df_demeaned[col] - df_demeaned.groupby('county')[col].transform('mean')
    )

for age_group, group in df_demeaned.groupby('AGE'):

    print("\n" + "="*60)
    print(f"DEMEANED MODEL - AGE GROUP: {age_group}")
    print("Group size:", group.shape)

    X = group[selected_pollutants]
    y = group['AGE_ADJUSTED_ED_VISIT_RATE']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        tree_method='hist',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² (demeaned): {r2:.4f}")
    print(f"RMSE (demeaned): {rmse:.4f}")

    plt.figure(figsize=(9, 5))
    xgb.plot_importance(model, importance_type='gain')
    plt.title(f"Demeaned Feature Importance ({age_group})")

    plt.tight_layout()
    save_path = f"{root_dir}/Scripts/feature_importance_demeaned_{age_group}.png"
    #plt.savefig(save_path, bbox_inches='tight')
    #plt.close()
    plt.show()

    print(f"Saved plot to: {save_path}")
    
    
    
# Part 3 - don't split by age, run one model using age as a feature
from sklearn.preprocessing import LabelEncoder

# 1. Prepare the unified dataset
df_unified = df_model.copy()

# Demean pollutants by county (keeping your existing logic)
for col in selected_pollutants:
    df_unified[col] = (
        df_unified[col] - df_unified.groupby('county')[col].transform('mean')
    )

# 2. Encode AGE (0 for adult, 1 for child)
le = LabelEncoder()
df_unified['AGE_encoded'] = le.fit_transform(df_unified['AGE'])

# 3. Define Features: Pollutants + Age
# We add AGE_encoded as a predictor!
features = selected_pollutants + ['AGE_encoded']
X = df_unified[features]
y = df_unified['AGE_ADJUSTED_ED_VISIT_RATE']

# 4. Split and Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=300, # Increased slightly
    learning_rate=0.03, # Lower learning rate often helps stability
    max_depth=5,       # Allow a bit more depth to capture age-pollutant interactions
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Unified Model R²: {r2:.4f}")
print(f"Unified Model RMSE: {rmse:.4f}")

# 6. Feature Importance (The "Why")
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='gain')
plt.title("Unified Feature Importance (Pollutants + Age)")
save_path = f"{root_dir}/figs/xgboost_features.png"
plt.savefig(save_path, bbox_inches='tight')
plt.show()


# shap explainer
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
save_path = f"{root_dir}/figs/xgboost_features.png"
plt.savefig(save_path, bbox_inches='tight')
