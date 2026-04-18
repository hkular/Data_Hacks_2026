#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:44:42 2026

@author: serenceslab
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Age Adjusted Rate of Asthma: number of asthma ED visits per 100,000 people in 
# geiven county, then adjusted by age.


# Load the excel file
file_path = "/Users/serenceslab/Desktop/Data_Hacks_2026/Asthma_Data/asthma-emergency-department-visit-rates-3s8pzn1q/asthma-ed-visit-rates-by-zip-code-and-age-groups-2013-present.csv"
df = pd.read_excel(file_path)

# Quick look at the data
print(df.head())  # First 5 rows
print(df.shape)   # Rows and columns
print(df.columns) # Column names

# Summary table
summary_table = df.groupby(['Year', 'County', 'Age_Group'])['Age_Adjusted_Rate_of_Asthma_ED_V'].mean().reset_index()
summary_table.columns = ['Year', 'County', 'Age_Group', 'Avg_Rate']
print(summary_table.head(10))

# Filter for children only
children_df = df[df['Age_Group'] == 'Child']

# Descriptive statistics by county
county_stats = children_df.groupby('County')['Age_Adjusted_Rate_of_Asthma_ED_V'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print("Child Asthma ED Visit Rates by County:")
print(county_stats)

# Visualize mean ED visits by county over time
county_means = children_df.groupby('County')['Age_Adjusted_Rate_of_Asthma_ED_V'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
county_means.plot(kind='barh')
plt.xlabel('Average Age-Adjusted Rate (per 100,000)')
plt.title('Child Asthma ED Visit Rates by County')
plt.tight_layout()
plt.show()

##############################################################################
# Linear Model  
##############################################################################


# Prepare data
model_df = df[['County', 'Year', 'Zip_Code', 'Age_Group', 'Age_Adjusted_Rate_of_Asthma_ED_V']].copy()
model_df = model_df.dropna()

# Encode categories
le_county = LabelEncoder()
le_age = LabelEncoder()
model_df['County'] = le_county.fit_transform(model_df['County'])
model_df['Age_encoded'] = le_age.fit_transform(model_df['Age_Group'])

X = model_df[['County', 'Year', 'Zip_Code', 'Age_encoded']]
y = model_df['Age_Adjusted_Rate_of_Asthma_ED_V']

model = LinearRegression()
model.fit(X, y)

print("Model: Rate = Intercept + County*coef + Year*coef + Zip_Code*coef + Age*coef")
print(f"\nIntercept: {model.intercept_:.2f}")
print(f"County coefficient: {model.coef_[0]:.4f}")
print(f"Year coefficient: {model.coef_[1]:.4f}")
print(f"Zip_Code coefficient: {model.coef_[2]:.4f}")
print(f"Age Group coefficient: {model.coef_[3]:.4f}")
print(f"R² (goodness of fit): {model.score(X, y):.3f}")