# -*- coding: utf-8 -*-
"""
Spyder Editor

Asthma data


"""

import pandas as pd
import numpy as np

root_path = 'Asthma_Data'

# er data
file_path = "Asthma_Data/asthma-ed-visit-rates-by-zip-code-and-age-groups-2013-present.csv"
df_er = pd.read_excel(file_path)

df_pc = pd.read_csv('Asthma_Data/chis-data-current-asthma-prevalence-by-county-2015-present.csv', encoding='latin-1')  
df_pl = pd.read_csv('Asthma_Data/chis-data-lifetime-asthma-prevalence-by-county-2015-present.csv', encoding='latin-1') 
df_d = pd.read_csv('Asthma_Data/asthma-deaths-by-county-2014-present.csv', encoding='latin-1')

# Quick look at the data
print(df_pc.columns)
print(df_pl.columns)
print(df_d.columns) 


# figure out where the data mismatchs
dfs = {'prevalence_current': df_pc, 'prevalence_lifetime': df_pl, 'deaths': df_d}
compare_cols = ['YEARS', 'COUNTY', 'AGE GROUP', 'STRATA']

for col in compare_cols:
    print(f"\n{'═'*50}")
    print(f"  {col}")
    print(f"{'═'*50}")
    
    sets = {}
    for name, df in dfs.items():
        if col in df.columns:
            sets[name] = set(np.unique(df[col]))
            print(f"\n  {name} ({len(sets[name])}):\n  {sorted(sets[name])}")
    
    # show what's unique to each df
    names = list(sets.keys())
    for name in names:
        others = set().union(*[sets[n] for n in names if n != name])
        unique_to = sets[name] - others
        if unique_to:
            print(f"\n  ** only in {name}: {sorted(unique_to)}")
    
    # show what's shared across all
    shared = set.intersection(*sets.values())
    print(f"\n  shared by all ({len(shared)}): {sorted(shared)}")
    
    
    
# fix year reporting
def clean_hyphen(year_str):
    # normalize the windows em-dash to a regular hyphen, then split
    clean = str(year_str).replace('\x96', '-')
    return str(clean)
for name, df in dfs.items():
    df['YEAR'] = df['YEARS'].apply(clean_hyphen)
    print(f"{name}: {sorted(df['YEAR'].unique())}")
    


# fix age group thing
def recode_age(df):
    # drop 'Total population' (aggregate over all ages)
    df = df[df['STRATA'] != 'Total population'].copy()
    
    # recode AGE GROUP to child/adult based on strata
    age_map = {
        '0\x9617 years': 'child',
        '0\x964 years':  'child',
        '5\x9617 years': 'child',
        '18+ years':     'adult',
        '18\x9664 years':'adult',
        '65+ years':     'adult',
        'All ages':       None,   # will drop below
    }
    
    df['AGE'] = df['AGE GROUP'].map(age_map)
    
    # drop 'All ages' rows (aggregate)
    df = df[df['AGE'].notna()].copy()
    
    # drop now-redundant columns
    df = df.drop(columns=['STRATA', 'AGE GROUP'])
    
    return df

df_d  = recode_age(df_d)
df_pc = recode_age(df_pc)
df_pl = recode_age(df_pl)
dfs = {'prevalence_county': df_pc, 'prevalence_local': df_pl, 'deaths': df_d}
# verify
for name, df in dfs.items():
    print(f"\n{name}")
    print(df['AGE'].unique())
    print(df.shape)



df_er = df_er.groupby(['Year', 'County', 'Age_Group']).agg(
    NUMBER_OF_ED_VISITS=('Number_of_Asthma_ED_Visits', 'sum'),
    AGE_ADJUSTED_ED_VISIT_RATE=('Age_Adjusted_Rate_of_Asthma_ED_V', 'mean')
).reset_index()

# rename to match other dataframes
df_er = df_er.rename(columns={
    'Year': 'YEAR',
    'County': 'COUNTY',
    'Age_Group': 'AGE GROUP'
})

print(df_er.shape)
print(df_er.head())
print(df_er['AGE GROUP'].unique())
df_er['AGE'] = df_er['AGE GROUP'].str.lower()
df_er = df_er.drop(columns=['AGE GROUP'])



df_er.to_csv('Asthma_Data/cleaned_er.csv', index=False)
df_d.to_csv('Asthma_Data/cleaned_deaths.csv', index=False)
df_pc.to_csv('Asthma_Data/cleaned_prevalence.csv', index=False)

print("Saved.")