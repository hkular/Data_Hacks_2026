# Simple linear regression on whether AQI and the different pollutants contribute to asthma-related ER
# visits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#########################################################################################################
# Load data, merge and run simple linear regression
#########################################################################################################

# Loaction of daily detailed particle datasets (in my local directory, not in github)
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
                            
    print(df_wide)

    # Start running regression across age groups
    for name, group in df_wide.groupby('AGE'):
        model = LinearRegression()
        
        X = group[select_particle]
        y = group['AGE_ADJUSTED_ED_VISIT_RATE']
        
        model.fit(X, y)
        
        print(f"Age Group: {name}")
        for col, coef in zip(select_particle, model.coef_):
            print(f"  {col} Coefficient: {coef:.4f}")
        print(f"  Intercept: {model.intercept_:.4f}")
        print(f"  R-squared: {model.score(X, y):.4f}")  







