import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load AQI data
print("Loading AQI data...")
aqi_files = [
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2015.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2016.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2017.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2018.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2019.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2020.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2021.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2022.csv',
    'EPA_data/air_aqi_annual/annual_aqi_by_county_2023.csv',
]

aqi_dfs = []
for file in aqi_files:
    try:
        df = pd.read_csv(file)
        aqi_dfs.append(df)
        print(f"  Loaded {file}")
    except Exception as e:
        print(f"  Error loading {file}: {e}")

aqi_df = pd.concat(aqi_dfs, ignore_index=True)
print(f"\nAQI data shape: {aqi_df.shape}")
print(f"AQI columns: {aqi_df.columns.tolist()}\n")

# Load ED visit rates by county with correct encoding
print("Loading ED visit rates by county...")
ed_df = pd.read_csv('Asthma_Data/asthma-emergency-department-visit-rates-3s8pzn1q/asthma-ed-visit-rates-by-county-2015-present.csv',
                     encoding='latin-1')
print(f"ED visits data shape: {ed_df.shape}")
print(f"ED visits columns: {ed_df.columns.tolist()}\n")

# Rename columns to match for merging
aqi_df = aqi_df.rename(columns={'County': 'COUNTY'})
ed_df = ed_df.rename(columns={'YEAR': 'Year'})

# Merge data by County and Year
print("Merging data...")
merged_df = aqi_df.merge(ed_df, on=['COUNTY', 'Year'], how='inner')
print(f"Merged data shape: {merged_df.shape}")
print(f"Unique counties in merged data: {merged_df['COUNTY'].nunique()}")
print(f"Year range: {merged_df['Year'].min()} - {merged_df['Year'].max()}\n")

# Display merged sample
print("Merged sample:")
print(merged_df[['COUNTY', 'Year', 'Median AQI', 'AGE-ADJUSTED ED VISIT RATE']].head(10))
print()

# Prepare data for linear regression
aqi_feature = merged_df[['Median AQI']].values
target = merged_df['AGE-ADJUSTED ED VISIT RATE'].values

# Remove any NaN values
valid_idx = ~(np.isnan(aqi_feature.flatten()) | np.isnan(target))
X = aqi_feature[valid_idx]
y = target[valid_idx]

print(f"Training set size: {len(X)} observations (after removing NaN)")
print(f"Feature (Median AQI) range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Target (Age-Adjusted ED Visit Rate) range: [{y.min():.2f}, {y.max():.2f}]\n")

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate metrics
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("=" * 70)
print("LINEAR REGRESSION RESULTS")
print("=" * 70)
print(f"Coefficient (slope): {model.coef_[0]:.6f}")
print(f"Intercept: {model.intercept_:.6f}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print()
print("Model Interpretation:")
print(f"  - For each unit increase in Median AQI,")
print(f"    age-adjusted ER visits change by {model.coef_[0]:.4f} visits")
print(f"  - This model explains {r2*100:.2f}% of the variance")
print()
print("Model equation:")
print(f"Age-Adjusted ER Visits = {model.intercept_:.4f} + {model.coef_[0]:.6f} * Median AQI")
print("=" * 70)

# Create visualization
plt.figure(figsize=(12, 7))
plt.scatter(X, y, alpha=0.5, s=50, label='County-Year observations')
plt.plot(X, y_pred, color='red', linewidth=2.5, label='Fitted regression line')
plt.xlabel('Median AQI', fontsize=12)
plt.ylabel('Age-Adjusted ED Visit Rate', fontsize=12)
plt.title(f'Predicting Age-Adjusted ER Visits from Annual Median AQI\n(R² = {r2:.4f})', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aqi_er_visits_linear_model.png', dpi=150)
print("\nPlot saved as 'aqi_er_visits_linear_model.png'")
