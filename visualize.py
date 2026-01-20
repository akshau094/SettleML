import pandas as pd
import os

# Load the merged data
file_path = r"c:\Users\aksha\OneDrive\Desktop\data for project\merged_settlement_data.csv"
df = pd.read_csv(file_path)

# 1. Top 10 Growing Districts (Based on Enrolments + Demographic Updates)
# We sum across all months to see the overall growth
district_growth = df.groupby(['state', 'district']).agg({
    'age_18_greater': 'sum',
    'demo_age_17_': 'sum',
    'bio_age_17_': 'sum'
}).reset_index()

district_growth['total_growth_score'] = district_growth['age_18_greater'] + district_growth['demo_age_17_']
top_growing = district_growth.sort_values(by='total_growth_score', ascending=False).head(10)

print("--- TOP 10 GROWING DISTRICTS (BOOM TOWNS) ---")
print(top_growing[['state', 'district', 'total_growth_score']])

# 2. Top 10 Districts for Family Settlements (Based on Children's Biometrics & Enrolments)
family_settlement = df.groupby(['state', 'district']).agg({
    'age_0_5': 'sum',
    'bio_age_5_17': 'sum',
    'demo_age_5_17': 'sum'
}).reset_index()

family_settlement['family_score'] = family_settlement['age_0_5'] + family_settlement['bio_age_5_17']
top_family = family_settlement.sort_values(by='family_score', ascending=False).head(10)

print("\n--- TOP 10 DISTRICTS FOR FAMILY SETTLEMENTS ---")
print(top_family[['state', 'district', 'family_score']])

# 3. Monthly Trends (Sum of all activities per month)
monthly_trend = df.groupby('month_year').agg({
    'age_18_greater': 'sum',
    'demo_age_17_': 'sum',
    'bio_age_17_': 'sum'
}).reset_index()

print("\n--- MONTHLY ACTIVITY TRENDS ---")
print(monthly_trend)
