#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:44:42 2026

@author: serenceslab
"""

import pandas as pd

# Load the excel file
file_path = "/Users/serenceslab/Desktop/Data_Hacks_2026/Asthma_Data/asthma-emergency-department-visit-rates-3s8pzn1q/asthma-ed-visit-rates-by-zip-code-and-age-groups-2013-present.csv"

df = pd.read_excel(file_path)

# Quick look at the data
print(df.head())  # First 5 rows
print(df.shape)   # Rows and columns
print(df.columns) # Column names