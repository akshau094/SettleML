import pandas as pd
import glob
import os

def load_and_aggregate(pattern, group_cols, sum_cols, name):
    files = glob.glob(pattern)
    all_data = []
    for f in files:
        print(f"Reading {f}...")
        df = pd.read_csv(f)
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        # Extract Month-Year for trend analysis
        df['month_year'] = df['date'].dt.to_period('M')
        
        # Aggregate
        agg = df.groupby(['month_year', 'state', 'district', 'pincode'])[sum_cols].sum().reset_index()
        all_data.append(agg)
    
    final_df = pd.concat(all_data).groupby(['month_year', 'state', 'district', 'pincode'])[sum_cols].sum().reset_index()
    print(f"Finished processing {name}. Shape: {final_df.shape}")
    return final_df

# Paths
base_path = r"c:\Users\aksha\OneDrive\Desktop\data for project"
biometric_pattern = os.path.join(base_path, "api_data_aadhar_biometric", "api_data_aadhar_biometric", "*.csv")
demographic_pattern = os.path.join(base_path, "api_data_aadhar_demographic", "api_data_aadhar_demographic", "*.csv")
enrolment_pattern = os.path.join(base_path, "api_data_aadhar_enrolment", "api_data_aadhar_enrolment", "*.csv")

# Column definitions
bio_cols = ['bio_age_5_17', 'bio_age_17_']
demo_cols = ['demo_age_5_17', 'demo_age_17_']
enrol_cols = ['age_0_5', 'age_5_17', 'age_18_greater']

# Load and aggregate
print("Processing Biometric Data...")
df_bio = load_and_aggregate(biometric_pattern, ['month_year', 'state', 'district', 'pincode'], bio_cols, "Biometric")

print("\nProcessing Demographic Data...")
df_demo = load_and_aggregate(demographic_pattern, ['month_year', 'state', 'district', 'pincode'], demo_cols, "Demographic")

print("\nProcessing Enrolment Data...")
df_enrol = load_and_aggregate(enrolment_pattern, ['month_year', 'state', 'district', 'pincode'], enrol_cols, "Enrolment")

# Merge all three
print("\nMerging datasets...")
df_final = df_bio.merge(df_demo, on=['month_year', 'state', 'district', 'pincode'], how='outer')
df_final = df_final.merge(df_enrol, on=['month_year', 'state', 'district', 'pincode'], how='outer')

# Fill NaNs with 0
df_final = df_final.fillna(0)

# Save the merged dataset
output_path = os.path.join(base_path, "merged_settlement_data.csv")
df_final.to_csv(output_path, index=False)
print(f"\nSuccess! Merged data saved to: {output_path}")
