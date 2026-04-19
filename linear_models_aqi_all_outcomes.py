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
aqi_df = aqi_df.rename(columns={'County': 'COUNTY', 'Year': 'YEAR'})
print(f"AQI data shape: {aqi_df.shape}\n")

# Load outcome datasets
print("Loading outcome datasets...")
er_df = pd.read_csv('cleaned_er.csv')
deaths_df = pd.read_csv('cleaned_deaths.csv')
prev_df = pd.read_csv('cleaned_prevalence.csv')

print(f"ER visits: {er_df.shape}")
print(f"Deaths: {deaths_df.shape}")
print(f"Prevalence: {prev_df.shape}\n")

# Convert YEAR to numeric for deaths and prevalence (extract first year from range)
# e.g., "2014-2016" -> 2014
def extract_first_year(year_str):
    if isinstance(year_str, str):
        return int(year_str.split('-')[0])
    return year_str

deaths_df['YEAR'] = deaths_df['YEAR'].apply(extract_first_year)
prev_df['YEAR'] = prev_df['YEAR'].apply(extract_first_year)
er_df['YEAR'] = er_df['YEAR'].astype(int)

# Function to fit and report model
def fit_and_report_model(aqi_data, outcome_data, outcome_name, target_col, age_group_name):
    print("=" * 75)
    print(f"MODEL: {outcome_name.upper()} - {age_group_name.upper()}")
    print("=" * 75)

    # Merge data
    merged_df = aqi_data.merge(outcome_data, on=['COUNTY', 'YEAR'], how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Unique counties: {merged_df['COUNTY'].nunique()}\n")

    # Prepare features and target
    X = merged_df[['Median AQI']].values
    y = merged_df[target_col].values

    # Remove NaN and infinite values
    valid_idx = ~(np.isnan(X.flatten()) | np.isnan(y) | np.isinf(y))
    X = X[valid_idx]
    y = y[valid_idx]

    if len(X) == 0:
        print("No valid data points after cleaning.\n")
        return None, None, None, None, None, None

    print(f"Training set size: {len(X)} observations")
    print(f"Feature (Median AQI) range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Target ({target_col}) range: [{y.min():.2f}, {y.max():.2f}]\n")

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Print results
    print("REGRESSION RESULTS:")
    print(f"  Coefficient: {model.coef_[0]:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

    direction = "increases" if model.coef_[0] > 0 else "decreases"
    print(f"For each unit increase in Median AQI, {target_col.lower()}")
    print(f"{direction} by {abs(model.coef_[0]):.4f}")
    print(f"Model explains {r2*100:.2f}% of variance\n")

    print("=" * 75 + "\n")

    return model, X, y, y_pred, r2, rmse

# ============================================================================
# ER VISITS MODELS
# ============================================================================
print("\n" + "█" * 75)
print("ER VISITS MODELS")
print("█" * 75 + "\n")

er_children = er_df[er_df['AGE'] == 'child'].copy()
er_adults = er_df[er_df['AGE'] == 'adult'].copy()

er_child_model, er_child_X, er_child_y, er_child_pred, er_child_r2, er_child_rmse = \
    fit_and_report_model(aqi_df, er_children, "ER Visits", "AGE_ADJUSTED_ED_VISIT_RATE", "Children")

er_adult_model, er_adult_X, er_adult_y, er_adult_pred, er_adult_r2, er_adult_rmse = \
    fit_and_report_model(aqi_df, er_adults, "ER Visits", "AGE_ADJUSTED_ED_VISIT_RATE", "Adults")

# ============================================================================
# DEATHS MODELS
# ============================================================================
print("\n" + "█" * 75)
print("MORTALITY MODELS")
print("█" * 75 + "\n")

deaths_children = deaths_df[deaths_df['AGE'] == 'child'].copy()
deaths_adults = deaths_df[deaths_df['AGE'] == 'adult'].copy()

death_child_model, death_child_X, death_child_y, death_child_pred, death_child_r2, death_child_rmse = \
    fit_and_report_model(aqi_df, deaths_children, "Mortality", "AGE-ADJUSTED MORTALITY RATE", "Children")

death_adult_model, death_adult_X, death_adult_y, death_adult_pred, death_adult_r2, death_adult_rmse = \
    fit_and_report_model(aqi_df, deaths_adults, "Mortality", "AGE-ADJUSTED MORTALITY RATE", "Adults")

# ============================================================================
# PREVALENCE MODELS
# ============================================================================
print("\n" + "█" * 75)
print("PREVALENCE MODELS")
print("█" * 75 + "\n")

prev_children = prev_df[prev_df['AGE'] == 'child'].copy()
prev_adults = prev_df[prev_df['AGE'] == 'adult'].copy()

prev_child_model, prev_child_X, prev_child_y, prev_child_pred, prev_child_r2, prev_child_rmse = \
    fit_and_report_model(aqi_df, prev_children, "Prevalence", "CURRENT PREVALENCE", "Children")

prev_adult_model, prev_adult_X, prev_adult_y, prev_adult_pred, prev_adult_r2, prev_adult_rmse = \
    fit_and_report_model(aqi_df, prev_adults, "Prevalence", "CURRENT PREVALENCE", "Adults")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Children
if er_child_model:
    axes[0, 0].scatter(er_child_X, er_child_y, alpha=0.5, s=30, color='blue')
    axes[0, 0].plot(er_child_X, er_child_pred, color='red', linewidth=2)
    axes[0, 0].set_title(f'ER Visits (Children)\nR² = {er_child_r2:.4f}', fontsize=11)
    axes[0, 0].set_xlabel('Median AQI')
    axes[0, 0].set_ylabel('ED Visit Rate')
    axes[0, 0].grid(True, alpha=0.3)

if death_child_model:
    axes[0, 1].scatter(death_child_X, death_child_y, alpha=0.5, s=30, color='blue')
    axes[0, 1].plot(death_child_X, death_child_pred, color='red', linewidth=2)
    axes[0, 1].set_title(f'Mortality (Children)\nR² = {death_child_r2:.4f}', fontsize=11)
    axes[0, 1].set_xlabel('Median AQI')
    axes[0, 1].set_ylabel('Mortality Rate')
    axes[0, 1].grid(True, alpha=0.3)

if prev_child_model:
    axes[0, 2].scatter(prev_child_X, prev_child_y, alpha=0.5, s=30, color='blue')
    axes[0, 2].plot(prev_child_X, prev_child_pred, color='red', linewidth=2)
    axes[0, 2].set_title(f'Prevalence (Children)\nR² = {prev_child_r2:.4f}', fontsize=11)
    axes[0, 2].set_xlabel('Median AQI')
    axes[0, 2].set_ylabel('Prevalence')
    axes[0, 2].grid(True, alpha=0.3)

# Row 2: Adults
if er_adult_model:
    axes[1, 0].scatter(er_adult_X, er_adult_y, alpha=0.5, s=30, color='green')
    axes[1, 0].plot(er_adult_X, er_adult_pred, color='red', linewidth=2)
    axes[1, 0].set_title(f'ER Visits (Adults)\nR² = {er_adult_r2:.4f}', fontsize=11)
    axes[1, 0].set_xlabel('Median AQI')
    axes[1, 0].set_ylabel('ED Visit Rate')
    axes[1, 0].grid(True, alpha=0.3)

if death_adult_model:
    axes[1, 1].scatter(death_adult_X, death_adult_y, alpha=0.5, s=30, color='green')
    axes[1, 1].plot(death_adult_X, death_adult_pred, color='red', linewidth=2)
    axes[1, 1].set_title(f'Mortality (Adults)\nR² = {death_adult_r2:.4f}', fontsize=11)
    axes[1, 1].set_xlabel('Median AQI')
    axes[1, 1].set_ylabel('Mortality Rate')
    axes[1, 1].grid(True, alpha=0.3)

if prev_adult_model:
    axes[1, 2].scatter(prev_adult_X, prev_adult_y, alpha=0.5, s=30, color='green')
    axes[1, 2].plot(prev_adult_X, prev_adult_pred, color='red', linewidth=2)
    axes[1, 2].set_title(f'Prevalence (Adults)\nR² = {prev_adult_r2:.4f}', fontsize=11)
    axes[1, 2].set_xlabel('Median AQI')
    axes[1, 2].set_ylabel('Prevalence')
    axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Linear Regression: AQI vs Asthma Outcomes (ER Visits, Mortality, Prevalence)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('aqi_all_outcomes_models.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'aqi_all_outcomes_models.png'")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 100)
print("COMPREHENSIVE SUMMARY TABLE")
print("=" * 100)
summary_data = {
    'Outcome': ['ER Visits', 'ER Visits', 'Mortality', 'Mortality', 'Prevalence', 'Prevalence'],
    'Age Group': ['Children', 'Adults', 'Children', 'Adults', 'Children', 'Adults'],
    'N': [
        len(er_child_X) if er_child_model else 0,
        len(er_adult_X) if er_adult_model else 0,
        len(death_child_X) if death_child_model else 0,
        len(death_adult_X) if death_adult_model else 0,
        len(prev_child_X) if prev_child_model else 0,
        len(prev_adult_X) if prev_adult_model else 0,
    ],
    'Coefficient': [
        er_child_model.coef_[0] if er_child_model else np.nan,
        er_adult_model.coef_[0] if er_adult_model else np.nan,
        death_child_model.coef_[0] if death_child_model else np.nan,
        death_adult_model.coef_[0] if death_adult_model else np.nan,
        prev_child_model.coef_[0] if prev_child_model else np.nan,
        prev_adult_model.coef_[0] if prev_adult_model else np.nan,
    ],
    'R² Score': [er_child_r2, er_adult_r2, death_child_r2, death_adult_r2, prev_child_r2, prev_adult_r2],
    'RMSE': [er_child_rmse, er_adult_rmse, death_child_rmse, death_adult_rmse, prev_child_rmse, prev_adult_rmse],
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("=" * 100)
