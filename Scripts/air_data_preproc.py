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
root_dir = '/mnt/neurocube/local/serenceslab/Stella/misc/Data_Hacks_2026'
in_dir = '/mnt/neurocube/local/serenceslab/Stella/misc/EPA_data'

# Years to screen (file name)
years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']

# Particles to screen (file name)
particles = ['CO', 'HAPS', 'lead', 'NO2', 'NONO', 'ozone', 'SO2', 'VOCS']

# Loop through the different years and particles
for year in years:
    for particle in particles:

        print(f"Processing {year} and {particle}...")

        # Load dataset 
        file_name = f'{in_dir}/air_{particle}_daily/*_{year}.csv'
        file = glob.glob(file_name)
        df = pd.read_csv(file[0], low_memory=False)

        # Only keep California entries
        df = df[df['State Name'] == 'California']

        # Calculate the mean, variance and percentiles of the measured particle concentration 
        df_compiled = df.groupby('County Name', as_index=False).agg(
            county=('County Name', 'first'),
            city=('City Name', 'first'),
            particle=('Parameter Name', 'first'),
            measurement=('Units of Measure', 'first'),
            days_recorded=('Date Local', 'nunique'),
            avg_val=('Arithmetic Mean', 'mean'),
            variance=('Arithmetic Mean', 'var'),
            min_val=('Arithmetic Mean', 'min'),
            percentile_10=('Arithmetic Mean', lambda x: x.quantile(0.1)), 
            percentile_25=('Arithmetic Mean', lambda x: x.quantile(0.25)), 
            percentile_75=('Arithmetic Mean', lambda x: x.quantile(0.75)), 
            percentile_90=('Arithmetic Mean', lambda x: x.quantile(0.9)), 
            max_val=('Arithmetic Mean', 'max')
        )
        if 'df_final' not in globals():
            df_final = df_compiled

        # Append year to the datafile
        df_compiled.insert(0, 'year', year)

        # Update 
        df_final = pd.concat([df_final, df_compiled], ignore_index=True)

# Save compiled output file
out_dir = f'{root_dir}/EPA_data/air_aqi_and_particles_annual/'
df_final.to_csv(f'{out_dir}/2015-2022.csv')
