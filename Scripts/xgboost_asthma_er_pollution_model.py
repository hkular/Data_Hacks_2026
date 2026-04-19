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

# Load all relevant data

root_dir = '/mnt/neurocube/local/serenceslab/isa/Data_Hacks_2026'

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
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


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
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to: {save_path}")