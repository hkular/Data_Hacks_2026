# Now that we know the list of candidates from the simple linear regression, we can move on to include
# all of them in the multivariate version.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

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

# Run multivariate regression, also include age as a categorical variable
formula = 'AGE_ADJUSTED_ED_VISIT_RATE ~ Q("Ozone") * C(AGE) + Q("Nitric oxide (NO)") * C(AGE)' + \
        '+ Q("Nitrogen dioxide (NO2)") * C(AGE)'
model = smf.ols(formula=formula, data=df_centered).fit()
print(model.summary())  
