# Simple linear regression on whether AQI and the different pollutants contribute to asthma-related ER
# visits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#########################################################################################################
# Load data, merge and run simple linear regression
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

# Realized that if we include all particles in our equation, there would always be NaNs for some counties
# or years in one of the particles, so first just test one by one with available data
for particle in particles:
    select_particle = [particle]
    subset_particle = subset.loc[subset['particle']==select_particle[0] ,:]

    # Reformat into wide format
    df_wide = subset_particle.pivot_table(
        index=['year', 'county', 'AGE', 'AGE_ADJUSTED_ED_VISIT_RATE'], 
        columns='particle', 
        values='avg_val'
    ).reset_index()

    # Start running regression across age groups
    for name, group in df_wide.groupby('AGE'):
        
        # Realized that since the county effect is probably large for some pollutants, and we cannot include county as a regressor
        # due to only 1 datapoint per age group per county, we have to implicitly remove its effect by mean-normalizing the pollutant
        # concentration within each county, then run regression. So slope interpretation is now: with every additional unit of pollutant
        # concentration with respect to the mean of a given county, what's the change in ED visit rate with respect to the mean in that
        # same county. 

        group['y_demeaned'] = group['AGE_ADJUSTED_ED_VISIT_RATE'] - group.groupby('county')['AGE_ADJUSTED_ED_VISIT_RATE'].transform('mean')
        group['X_demeaned'] = group[select_particle] - group.groupby('county')[select_particle].transform('mean')
        group['X_mean'] = group.groupby('county')[select_particle].transform('mean') # also save the mean particle concentration itself

        X = group['X_demeaned']
        y = group['y_demeaned']

        # For demeaned data, intercept will be 0 so no need to include, but let's just have it here anyways in case we want to
        # go back to using the original raw data
        X = sm.add_constant(X)
        sm_model = sm.OLS(y, X).fit()

        slope = sm_model.params.iloc[1]
        pval = sm_model.pvalues.iloc[1]
        R2 = sm_model.rsquared
        
        if pval < 0.05: print(f"!!! Signicant !!!")
        print(f"{select_particle[0]}")
        print(f"\tAge Group: {name}")
        print(f"\t\tSlope: {slope:.4f}")
        print(f"\t\tP-value: {pval: .4f}")
        print(f"\t\tR-squared: {R2:.4f}")
        print(f"\t\tNum Datapoints: {len(y)}\n")

        # An overview of the average pollutant concentration across counties 
        print(f"\t\tMean Pollutant Concentration: {group[['county', 'X_mean']].drop_duplicates()}\n\n")
        print("="*70)






