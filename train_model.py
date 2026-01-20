import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load the merged data
file_path = r"c:\Users\aksha\OneDrive\Desktop\data for project\merged_settlement_data.csv"
df = pd.read_csv(file_path)

# 1. Feature Engineering per Pincode
pincode_features = df.groupby(['state', 'district', 'pincode']).agg({
    'age_18_greater': ['sum', 'mean', 'std'],
    'demo_age_17_': ['sum', 'mean', 'std'],
    'bio_age_5_17': ['sum', 'mean'],
    'age_0_5': 'sum'
}).reset_index()

# Flatten columns robustly
pincode_features.columns = [f"{col[0]}_{col[1]}".strip('_') for col in pincode_features.columns.values]

# Fill NaNs
pincode_features = pincode_features.fillna(0)

# Create simplified scores (Check for double underscores if they exist)
# We use a safer way to find columns
def get_col(prefix, suffix):
    for c in pincode_features.columns:
        if c.startswith(prefix) and c.endswith(suffix):
            return c
    return None

arrival_col = get_col('age_18_greater', 'sum')
stay_intent_col = get_col('demo_age_17', 'sum')
family_bio_col = get_col('bio_age_5_17', 'sum')
family_enrol_col = get_col('age_0_5', 'sum')

pincode_features['arrival_score'] = pincode_features[arrival_col]
pincode_features['stay_intent_score'] = pincode_features[stay_intent_col]
pincode_features['family_settlement_score'] = pincode_features[family_bio_col] + pincode_features[family_enrol_col]

# Select features for clustering
X = pincode_features[['arrival_score', 'stay_intent_score', 'family_settlement_score']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train KMeans Model
print("Training KMeans clustering model...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
pincode_features['cluster'] = kmeans.fit_predict(X_scaled)

# 3. Label the clusters based on their characteristics
# We identify which cluster is "Growing" by looking at the mean scores
cluster_means = pincode_features.groupby('cluster')[['arrival_score', 'stay_intent_score', 'family_settlement_score']].mean()
print("\nCluster Characteristics:")
print(cluster_means)

# Mapping: Highest stay_intent_score usually means "Growing"
growing_cluster = cluster_means['stay_intent_score'].idxmax()
# Lowest usually means "Low Activity"
low_activity_cluster = cluster_means['stay_intent_score'].idxmin()
# The one in the middle is "Stable"
all_clusters = [0, 1, 2]
stable_cluster = [c for c in all_clusters if c not in [growing_cluster, low_activity_cluster]][0]

cluster_map = {
    growing_cluster: "Boom Town (Growing)",
    stable_cluster: "Stable Area",
    low_activity_cluster: "Low Activity Area"
}

pincode_features['settlement_type'] = pincode_features['cluster'].map(cluster_map)

# 4. Save the results
output_path = r"c:\Users\aksha\OneDrive\Desktop\data for project\pincode_settlement_predictions.csv"
pincode_features.to_csv(output_path, index=False)

print(f"\nModel training complete! Results saved to: {output_path}")

# Display some examples
print("\n--- SAMPLE PREDICTIONS ---")
print(pincode_features[['state', 'district', 'pincode', 'settlement_type']].sample(15))

# Count of each type
print("\n--- SUMMARY OF PINCODES ---")
print(pincode_features['settlement_type'].value_counts())
