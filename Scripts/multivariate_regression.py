# Now that we know the list of candidates from the simple linear regression, we can move on to include
# all of them in the multivariate version.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

#########################################################################################################
# Load data, merge and run multivariate regression
#########################################################################################################

# Data directories for load and save
root_dir = '/mnt/neurocube/local/serenceslab/Stella/misc/Data_Hacks_2026'
out_dir = '/mnt/neurocube/local/serenceslab/Stella/misc/Data_Hacks_2026/Figs'

# EPA air quality data
air_data = pd.read_csv(f'{root_dir}/EPA_data/air_aqi_and_particles_annual/2015-2022.csv')
med_data = pd.read_csv(f'{root_dir}/Asthma_Data/cleaned_er.csv')

# Merge the two datasets
merge_data = pd.merge(air_data, med_data, 
                      left_on=['year','county'], right_on=['YEAR','COUNTY'])

# Categories for stratification
counties = merge_data['county'].unique().tolist()
age_groups = ['adult', 'child']

# Predictive variables
particles = merge_data['particle'].unique().tolist()

# Keep only necessary columns
subset = merge_data.loc[:, ['year', 'county', 'AGE', 'AGE_ADJUSTED_ED_VISIT_RATE', 'particle', 'avg_val']]

# Only include pollutant candidates that are relevant and have large enough number of datapoints 
select_particles = ['Nitrogen dioxide (NO2)', 'Nitric oxide (NO)', 'Ozone'] # removed SO2 due to still too few datapoints
subset = subset.loc[(subset['particle']).isin(select_particles) ,:]

# Reformat into wide format
df_wide = subset.pivot_table(
    index=['year', 'county', 'AGE', 'AGE_ADJUSTED_ED_VISIT_RATE'], 
    columns='particle', 
    values='avg_val'
).reset_index()

# Save the baseline average ED visit rate for each county, put back later during prediction (assuming it doesn't
# change significantly across years)
county_means = df_wide.groupby('county')['AGE_ADJUSTED_ED_VISIT_RATE'].mean().reset_index()
county_means.columns = ['county', 'baseline_mean']
# Need them for the pollutants too or else cannot demean lateset data
historical_pollutant_means = df_wide.groupby('county')[select_particles].mean().reset_index()

# Again, de-mean pollutants and ED visit rates within each county
cols_to_demean = select_particles + ['AGE_ADJUSTED_ED_VISIT_RATE']
df_centered = df_wide.copy()
df_centered[cols_to_demean] = df_centered.groupby('county')[cols_to_demean].transform(lambda x: x - x.mean())

# Still lots of NaNs, impute missing values with the mean of pollutant concentration in that county across years
for particle in select_particles:
    df_centered[particle] = df_centered.groupby('county')[particle].transform(lambda x: x.fillna(x.mean()))
print(f"After data imputation, still have to drop {df_centered.isna().any(axis=1).sum()} entries due to one or more"
        f"missing pollutant data.") 
df_final = df_centered.dropna()

# Run multivariate regression, also include age as a categorical variable (run on full data once just to get the
# overall stats, can print individual ones within cv if needed)
formula = 'AGE_ADJUSTED_ED_VISIT_RATE ~ Q("Ozone") * C(AGE) + Q("Nitric oxide (NO)") * C(AGE)' + \
        '+ Q("Nitrogen dioxide (NO2)") * C(AGE)'
model = smf.ols(formula=formula, data=df_centered).fit()
print(model.summary())  


#########################################################################################################
# Test prediction and visualize the regression goodness of fit
#########################################################################################################

# Create new column with the predicted ED rates
df_final = df_final.copy()
df_final['predicted_er_rate'] = np.nan

# Split data and CV when running model
kf = KFold(n_splits=20, shuffle=True, random_state=411)
fold_r2 = []
fold_mse = []
for train_index, test_index in kf.split(df_final):
    
    train_data = df_final.iloc[train_index]
    test_data = df_final.iloc[test_index]

    fold_model = smf.ols(formula=formula, data=train_data).fit()
    
    # Save predictions
    predicted = fold_model.predict(test_data)
    df_final.loc[df_final.index[test_index], 'predicted_er_rate'] = predicted

    # Save metrics for current fold
    r2 = r2_score(test_data['AGE_ADJUSTED_ED_VISIT_RATE'], predicted)
    mse = mean_squared_error(test_data['AGE_ADJUSTED_ED_VISIT_RATE'], predicted)
    fold_r2.append(r2)
    fold_mse.append(mse)
    print(f"This fold: R² = {r2:.3f}, MSE = {mse:.3f}\n")

# Average and spread of model fit metrics
mean_r2 = np.mean(fold_r2)
std_r2 = np.std(fold_r2)
mean_mse = np.mean(fold_mse)
std_mse = np.std(fold_mse)
print("============= Final CV Performance =============")
print(f"Mean R²:  {mean_r2:.3f} ± {std_r2:.3f}")
print(f"Mean MSE: {mean_mse:.3f} ± {std_mse:.3f}")

# Visualize results across all folds
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_final, x='AGE_ADJUSTED_ED_VISIT_RATE', y='predicted_er_rate', hue='AGE')

line_min = min(df_final['AGE_ADJUSTED_ED_VISIT_RATE'].min(), df_final['predicted_er_rate'].min())
line_max = max(df_final['AGE_ADJUSTED_ED_VISIT_RATE'].max(), df_final['predicted_er_rate'].max())
plt.plot([line_min, line_max], [line_min, line_max], color='red', linestyle='--', label='Ideal Perfect Fit')

plt.title(f"Cross-Validated Model Fit: Actual vs. Predicted (5 Folds)\n Mean R2 = {mean_r2}, Mean MSE = {mean_mse}")
plt.xlabel("Actual ER Visit Rate (Demeaned)")
plt.ylabel("Predicted ER Visit Rate")
plt.legend()
plt.savefig(f'{out_dir}/cv_prediction.png')
plt.show()



#########################################################################################################
# Now actually test on new data (air quality reports 2025) where ER visit data is not avilable 
#########################################################################################################

# Load 2025 air pollution data
new_air_data = pd.read_csv(f'{root_dir}/EPA_data/air_aqi_and_particles_annual/2023-2025.csv')

# Make sure to preprocess this dataset the same way we did when training model to maintain consistency
# and same interpretability, including demeaning each pollutant by within its county.
new_subset = new_air_data.loc[:, ['year', 'county', 'particle', 'avg_val']]
new_subset = new_subset.loc[(new_subset['particle']).isin(select_particles) ,:]
new_df_wide = new_subset.pivot_table(
    index=['year', 'county'], 
    columns='particle', 
    values='avg_val'
).reset_index()

# Again, de-mean pollutants within each county based on PAST data
new_df_centered = new_df_wide.copy()
new_df_centered = new_df_centered.merge(historical_pollutant_means, on='county', suffixes=('', '_hist'))
for particle in select_particles:
    new_df_centered[particle] = new_df_centered[particle] - new_df_centered[f'{particle}_hist']
# Drop helper columns used for demeaning, the historical content
new_df_centered = new_df_centered.drop(columns=[f'{p}_hist' for p in select_particles])

# Still lots of NaNs, impute missing values with the mean of pollutant concentration in that county across years
# If missing, find the most recent year in the file. Still this doesn't save states with no data in the past e
# years so they will just have to be empty. 
melted = new_df_centered.melt(
    id_vars=['year', 'county'], 
    value_vars=select_particles, 
    var_name='particle', 
    value_name='avg_val'
)
melted = melted.dropna(subset=['avg_val'])
idx = melted.groupby(['county', 'particle'])['year'].idxmax()
latest_data_points = melted.loc[idx]
new_df_final = latest_data_points.pivot(
    index='county', 
    columns='particle', 
    values='avg_val'
).reset_index()
new_df_final['reference_year'] = latest_data_points.groupby('county')['year'].max().values

print(f"After data imputation, still have to drop {new_df_centered.isna().any(axis=1).sum()} entries due to one or more"
        f"missing pollutant data.") 
new_df_final = new_df_final.dropna()

# Add new column of age group so model can predict accordingly
new_df_final['AGE'] = 'child'
copy_df_final = new_df_final.copy()
copy_df_final['AGE'] = 'adult'
final_test_df = pd.concat([new_df_final, copy_df_final], ignore_index=True)

# Predict and add back the mean
output = pd.DataFrame(columns=['county', 'age_group', 'risk_score'])
predicted = model.predict(final_test_df)
final_test_df.loc[final_test_df.index, 'predicted_diff'] = predicted

# Add back the baseline ED visit mean for each county back
final_test_df = final_test_df.merge(county_means, on='county', how='left')
final_test_df['final_predicted_er_rate'] = final_test_df['predicted_diff'] + final_test_df['baseline_mean']

# Reformat to be saved and pumped onto dashboard
output['county'] = final_test_df['county']
output['age_group'] = final_test_df['AGE']
output['risk_score'] = final_test_df['final_predicted_er_rate']
save_dir = f'{root_dir}/Asthma_Data/'
output.to_csv(f'{save_dir}/predicted.csv')

print(final_test_df)