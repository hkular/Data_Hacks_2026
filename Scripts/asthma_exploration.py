# -*- coding: utf-8 -*-
"""
Spyder Editor

Asthma data


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


df_er = pd.read_csv('/Volumes/serenceslab/holly/Data_Hacks_2026/Asthma_Data/cleaned_er.csv')
df_d = pd.read_csv('/Volumes/serenceslab/holly/Data_Hacks_2026/Asthma_Data/cleaned_deaths.csv')
df_p = pd.read_csv('/Volumes/serenceslab/holly/Data_Hacks_2026/Asthma_Data/cleaned_prevalence.csv')

df = df_er.copy()

# Quick look at the data
print(df.head())
print(df.shape)
print(df.columns)

# Summary table
summary_table = df.groupby(['YEAR', 'COUNTY', 'AGE'])['AGE_ADJUSTED_ED_VISIT_RATE'].mean().reset_index()
summary_table.columns = ['Year', 'County', 'Age_Group', 'Avg_Rate']
print(summary_table.head(10))

# Filter for children only
children_df = df[df['AGE'] == 'child']

# Descriptive statistics by county
county_stats = children_df.groupby('COUNTY')['AGE_ADJUSTED_ED_VISIT_RATE'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print("Child Asthma ED Visit Rates by County:")
print(county_stats)

# Visualize mean ED visits by county over time
county_means = children_df.groupby('COUNTY')['AGE_ADJUSTED_ED_VISIT_RATE'].mean().sort_values(ascending=False)
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
model_df = df[['COUNTY', 'YEAR', 'AGE', 'AGE_ADJUSTED_ED_VISIT_RATE']].copy()
model_df = model_df.dropna()

# Encode categories
le_county = LabelEncoder()
le_age = LabelEncoder()
model_df['COUNTY'] = le_county.fit_transform(model_df['COUNTY'])
model_df['AGE_encoded'] = le_age.fit_transform(model_df['AGE'])

X = model_df[['COUNTY', 'YEAR', 'AGE_encoded']]
y = model_df['AGE_ADJUSTED_ED_VISIT_RATE']

model = LinearRegression()
model.fit(X, y)
print("Model: Rate = Intercept + County*coef + Year*coef + Zip_Code*coef + Age*coef")
print(f"\nIntercept: {model.intercept_:.2f}")
print(f"County coefficient: {model.coef_[0]:.4f}")
print(f"Year coefficient: {model.coef_[1]:.4f}")
print(f"Zip_Code coefficient: {model.coef_[2]:.4f}")
print(f"Age Group coefficient: {model.coef_[3]:.4f}")
print(f"R² (goodness of fit): {model.score(X, y):.3f}")