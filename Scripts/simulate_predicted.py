#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:22:15 2026

@author: hollykular
"""

import pandas as pd
import numpy as np

# List of all 58 California Counties
counties = [
    "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa", 
    "Del Norte", "El Dorado", "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo", 
    "Kern", "Kings", "Lake", "Lassen", "Los Angeles", "Madera", "Marin", "Mariposa", 
    "Mendocino", "Merced", "Modoc", "Mono", "Monterey", "Napa", "Nevada", "Orange", 
    "Placer", "Plumas", "Riverside", "Sacramento", "San Benito", "San Bernardino", 
    "San Diego", "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo", 
    "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", "Sierra", "Siskiyou", 
    "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama", "Trinity", "Tulare", 
    "Tuolumne", "Ventura", "Yolo", "Yuba"
]

data = []

for county in counties:
    for age_group in ["adult", "child"]:
        # Generating a random risk score between 10 and 95
        risk_score = np.random.uniform(10, 95)
        data.append({
            "county": county,
            "age_group": age_group,
            "risk_score": round(risk_score, 2)
        })

# Create DataFrame
df_fake_predicted = pd.DataFrame(data)

# Save to the directory your dashboard expects
# Note: Ensure the 'Asthma_Data' folder exists first
import os
os.makedirs("../Asthma_Data", exist_ok=True)
df_fake_predicted.to_csv("../Asthma_Data/predicted.csv", index=False)

print("predicted.csv generated successfully with 116 rows.")