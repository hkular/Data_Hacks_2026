# Preprocessing script that takes in csv files on daily measured concentrations of different subcategories
# of pollutants, and compile the results into a single csv document for downstream analysis:
#    - Only keep data collected within California
#    - Derive the mean, variance, and 10 - 90th percentiles of the values across all days where data was
#      recorded
#    - Each year will have its own set of statistics 

import glob
import pandas as pd

#########################################################################################################
# Load data and preprocess
#########################################################################################################

# Loaction of daily detailed particle datasets (in my local directory, not in github)
root_dir = '/Volumes/serenceslab/holly/Data_Hacks_2026'#'/mnt/neurocube/local/serenceslab/holly/Data_Hacks_2026'
in_dir = '/Volumes/serenceslab/holly/Data_Hacks_2026/EPA_data'#'/mnt/neurocube/local/serenceslab/holly/Data_Hacks_2026/EPA_data'

# Years to screen (file name)
years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']

all_years = []

# Loop through the different years and particles
for year in years:
    
    print(f"Processing {year} and AQI...")
    # Load dataset 
    file_name = f'{in_dir}/air_aqi_daily/*_{year}.csv'
    file = glob.glob(file_name)
    df = pd.read_csv(file[0], low_memory=False)

    # Only keep California entries
    df = df[df['State Name'] == 'California']

    # Calculate the mean, variance and percentiles of the measured particle concentration 
    df_compiled = df.groupby('county Name', as_index=False).agg(
        county=('county Name', 'first'),
        definition = ('Defining Parameter', 'nunique'),
        aqi=('AQI','mean'),
        days_recorded=('Date', 'nunique'),
        avg_val=('AQI', 'mean'),
        variance=('AQI', 'var'),
        min_val=('AQI', 'min'),
        percentile_10=('AQI', lambda x: x.quantile(0.1)), 
        percentile_25=('AQI', lambda x: x.quantile(0.25)), 
        percentile_75=('AQI', lambda x: x.quantile(0.75)), 
        percentile_90=('AQI', lambda x: x.quantile(0.9)), 
        max_val=('AQI', 'max'),
        sites_avg=('Number of Sites Reporting', 'mean')
    )
    df_compiled.insert(0, 'year', year)
    all_years.append(df_compiled)

# Concatenate all years at once
df_final = pd.concat(all_years, ignore_index=True)

out_dir = f'{root_dir}/EPA_data/air_aqi_and_particles_annual/'
df_final.to_csv(f'{out_dir}/aqi_2015-2022.csv', index=False)