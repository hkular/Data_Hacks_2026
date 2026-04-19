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
    df = pd.read_csv(file)
    aqi_dfs.append(df)

aqi_df = pd.concat(aqi_dfs, ignore_index=True)
print(f"AQI data shape: {aqi_df.shape}\n")

# Load ED visit rates by county
print("Loading ED visit rates by county...")
ed_df = pd.read_csv('Asthma_Data/asthma-emergency-department-visit-rates-3s8pzn1q/asthma-ed-visit-rates-by-county-2015-present.csv',
                     encoding='latin-1')
print(f"ED visits data shape: {ed_df.shape}")
print(f"Age groups available: {ed_df['AGE GROUP'].unique()}\n")

# Rename columns to match for merging
aqi_df = aqi_df.rename(columns={'County': 'COUNTY'})
ed_df = ed_df.rename(columns={'YEAR': 'Year'})

# Define age groups
children_pattern = '0\x9617 years'  # 0-17 years
adults_pattern = '18+ years'         # 18+ years

# Split data by age group
ed_children = ed_df[ed_df['AGE GROUP'] == children_pattern].copy()
ed_adults = ed_df[ed_df['AGE GROUP'] == adults_pattern].copy()

print(f"Children data: {len(ed_children)} rows")
print(f"Adults data: {len(ed_adults)} rows\n")

# Function to create and evaluate model
def fit_and_report_model(aqi_data, ed_data, age_group_name):
    print("=" * 70)
    print(f"MODEL: {age_group_name.upper()}")
    print("=" * 70)

    # Merge data
    merged_df = aqi_data.merge(ed_data, on=['COUNTY', 'Year'], how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Unique counties: {merged_df['COUNTY'].nunique()}")
    print(f"Year range: {merged_df['Year'].min()} - {merged_df['Year'].max()}\n")

    # Prepare features and target
    X = merged_df[['Median AQI']].values
    y = merged_df['AGE-ADJUSTED ED VISIT RATE'].values

    # Remove NaN values
    valid_idx = ~(np.isnan(X.flatten()) | np.isnan(y))
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"Training set size: {len(X)} observations (after removing NaN)")
    print(f"Feature (Median AQI) range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Target (Age-Adjusted ED Visit Rate) range: [{y.min():.2f}, {y.max():.2f}]\n")

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Print results
    print("REGRESSION RESULTS:")
    print(f"  Coefficient (slope): {model.coef_[0]:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

    print("Model Interpretation:")
    direction = "increases" if model.coef_[0] > 0 else "decreases"
    print(f"  - For each unit increase in Median AQI,")
    print(f"    age-adjusted ER visits {direction} by {abs(model.coef_[0]):.4f} visits")
    print(f"  - This model explains {r2*100:.2f}% of the variance\n")

    print("Model Equation:")
    print(f"Age-Adjusted ED Visit Rate = {model.intercept_:.4f} + {model.coef_[0]:.6f} * Median AQI")
    print("=" * 70 + "\n")

    return model, X, y, y_pred, r2, rmse

# Fit both models
model_children, X_children, y_children, y_pred_children, r2_children, rmse_children = \
    fit_and_report_model(aqi_df, ed_children, "Children (0-17 years)")

model_adults, X_adults, y_adults, y_pred_adults, r2_adults, rmse_adults = \
    fit_and_report_model(aqi_df, ed_adults, "Adults (18+ years)")

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Children plot
axes[0].scatter(X_children, y_children, alpha=0.5, s=50, color='blue', label='Observations')
axes[0].plot(X_children, y_pred_children, color='red', linewidth=2.5, label='Fitted line')
axes[0].set_xlabel('Median AQI', fontsize=12)
axes[0].set_ylabel('Age-Adjusted ED Visit Rate', fontsize=12)
axes[0].set_title(f'Children (0-17 years)\nR² = {r2_children:.4f}, RMSE = {rmse_children:.2f}', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Adults plot
axes[1].scatter(X_adults, y_adults, alpha=0.5, s=50, color='green', label='Observations')
axes[1].plot(X_adults, y_pred_adults, color='red', linewidth=2.5, label='Fitted line')
axes[1].set_xlabel('Median AQI', fontsize=12)
axes[1].set_ylabel('Age-Adjusted ED Visit Rate', fontsize=12)
axes[1].set_title(f'Adults (18+ years)\nR² = {r2_adults:.4f}, RMSE = {rmse_adults:.2f}', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Linear Regression: AQI vs Age-Adjusted ED Visit Rates by Age Group',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('aqi_er_visits_by_age_group.png', dpi=150, bbox_inches='tight')
print("Comparison plot saved as 'aqi_er_visits_by_age_group.png'")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)
print(f"{'Metric':<30} {'Children (0-17)':<20} {'Adults (18+)':<20}")
print("-" * 70)
print(f"{'Number of observations':<30} {len(X_children):<20} {len(X_adults):<20}")
print(f"{'Coefficient':<30} {model_children.coef_[0]:<20.6f} {model_adults.coef_[0]:<20.6f}")
print(f"{'Intercept':<30} {model_children.intercept_:<20.6f} {model_adults.intercept_:<20.6f}")
print(f"{'R² Score':<30} {r2_children:<20.4f} {r2_adults:<20.4f}")
print(f"{'RMSE':<30} {rmse_children:<20.4f} {rmse_adults:<20.4f}")
print("=" * 70)
