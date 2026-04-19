#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:24:51 2026

@author: ilongoriavalenzuela
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re


# This script is just to get an idea of the variance in the AQI data

# Currently plots: 
# 1) California counties by good AQI days over time 
# 2) Pollutants in each california county over time

# ----------------------------
# config stuff
# ----------------------------
BASE_DIR = "/mnt/neurocube/local/serenceslab/isa"
data_path = os.path.join(BASE_DIR, "Data_Hacks_2026/EPA_data/air_aqi_annual/*.csv")

state_column = "State"
county_column = "County"

aqi_days_col = "Days with AQI"
good_days_col = "Good Days"

# ----------------------------
# load and combine all the yearly data into one big df
# FYI: Data range is from 2015- 2025
# ----------------------------
all_files = glob.glob(data_path)

dfs = []

for file in all_files:
    # Extract year from filename
    match = re.search(r"(20\d{2})", os.path.basename(file))
    if not match:
        continue

    year = int(match.group(1))

    df = pd.read_csv(file)
    df["Year"] = year

    # ----------------------------
    # get an aqi metric by looking at good days over total days of aqi data collected 
    # diff counties have diff number recorded, to standardize
    # ----------------------------
    # percent of good AQI days
    df["good_day_pct"] = (df[good_days_col] / df[aqi_days_col]) * 100
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

# ----------------------------
# filter for california only
# ----------------------------
ca_df = full_df[full_df[state_column].str.lower() == "california"]

# ----------------------------
# aggregate by year and county 
# ----------------------------
county_year = (
    ca_df
    .groupby(["County", "Year"])["good_day_pct"]
    .mean()
    .reset_index()
)

# ----------------------------
# 1) plot each county good AQI over time
# ----------------------------
plt.figure(figsize=(14, 8))

for county in county_year["County"].unique():
    subset = county_year[county_year["County"] == county].sort_values("Year")
    plt.plot(subset["Year"], subset["good_day_pct"], alpha=0.7, linewidth=1)

plt.title("California County AQI Quality Over Time")
plt.xlabel("Year")
plt.ylabel("Percent Good AQI Days")

# move legend outside plot area
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=6,
    frameon=False
)

plt.tight_layout()
plt.show()

# ----------------------------
# 2) plot variations in particles in AQI data 

# These are the ones collected: Co, NO2, Ozone (general), PM2.5, PM10
# From the dataset, it is unclear what the "Days" means, but I am gonna assume: 
# the number is how many days that pollutant was present / recorded / exceeded category

# Plots will be for each pollutant, separated by county, over time
# ----------------------------

pollutants = ["Days CO", "Days NO2", "Days Ozone", "Days PM2.5", "Days PM10"]


# use only california data again 
ca_df = ca_df.copy()

for pollutant in pollutants:

    plt.figure(figsize=(12, 7))

    for county in ca_df["County"].unique():

        subset = ca_df[ca_df["County"] == county].sort_values("Year")

        # skip if column missing or all NaN (should only be 0 though)
        if pollutant not in subset.columns:
            continue

        plt.plot(
            subset["Year"],
            subset[pollutant],
            marker="o",
            linewidth=1,
            alpha=0.7,
            label=county
        )

    plt.title(f"California Counties - {pollutant} Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Days")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6,
        frameon=False
    )

    plt.tight_layout()
    plt.show()

# ----------------------------
# 3) Is pollution "burden" for ALL counties in California changing over time? 

# using each pollutant instead of AQI (which uses dominant pollutant to report AQI)
# for now collapse across counties 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#  create a pollution index so that one doesn't dominante unfairly

# pollution index
pollutant_cols = ["Days CO", "Days NO2", "Days Ozone", "Days PM2.5", "Days PM10"]

#gets pollutant columns, years, and gets rid of any data rows that have no data
model_df = ca_df[pollutant_cols + ["Year"]].dropna()

#standardize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(model_df[pollutant_cols])

# index to get one number per row, overall pollution burden
# so higher = worse burden, lower = less burden 
model_df["pollution_index"] = X_scaled.sum(axis=1)

X = model_df[["Year"]] #predictor
y = model_df["pollution_index"] #outcome

# do the regression to get trend over the years
model = LinearRegression()
model.fit(X, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Year coefficient: {model.coef_[0]:.4f}")
print(f"R²: {model.score(X, y):.3f}")




