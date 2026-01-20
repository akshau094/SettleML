import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

# Load datasets
base_path = r"c:\Users\aksha\OneDrive\Desktop\data for project"
merged_data = pd.read_csv(os.path.join(base_path, "merged_settlement_data.csv"))
predictions = pd.read_csv(os.path.join(base_path, "pincode_settlement_predictions.csv"))

# Create a directory for diagrams
diagrams_path = os.path.join(base_path, "diagrams")
if not os.path.exists(diagrams_path):
    os.makedirs(diagrams_path)

# --- 1. Top 10 Growing Districts (Bar Chart) ---
plt.figure(figsize=(12, 6))
district_growth = merged_data.groupby('district')[['age_18_greater', 'demo_age_17_']].sum().sum(axis=1).sort_values(ascending=False).head(10)
sns.barplot(x=district_growth.values, y=district_growth.index, palette="viridis")
plt.title("Top 10 Boom Town Districts (Total Arrivals & Updates)", fontsize=15)
plt.xlabel("Total Count of Activity", fontsize=12)
plt.ylabel("District", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(diagrams_path, "top_growing_districts.png"))
plt.close()

# --- 2. Settlement Type Distribution (Pie Chart) ---
plt.figure(figsize=(8, 8))
type_counts = predictions['settlement_type'].value_counts()
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=140)
plt.title("Distribution of Pincode Settlement Types (AI Classification)", fontsize=15)
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(diagrams_path, "settlement_distribution.png"))
plt.close()

# --- 3. Monthly Activity Trends (Line Graph) ---
plt.figure(figsize=(12, 6))
# Convert month_year back to string for plotting if it's not already
merged_data['month_year'] = merged_data['month_year'].astype(str)
monthly_data = merged_data.groupby('month_year').agg({
    'age_18_greater': 'sum',
    'demo_age_17_': 'sum',
    'bio_age_5_17': 'sum'
}).reset_index()

sns.lineplot(data=monthly_data, x='month_year', y='age_18_greater', label='New Arrivals (Enrolment)', marker='o')
sns.lineplot(data=monthly_data, x='month_year', y='demo_age_17_', label='Staying Intent (Address Update)', marker='s')
sns.lineplot(data=monthly_data, x='month_year', y='bio_age_5_17', label='Family Stability (Biometric)', marker='^')

plt.title("The 'Migratory Pulse' - Monthly Trends Across India", fontsize=15)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of People", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(diagrams_path, "monthly_pulse_trends.png"))
plt.close()

# --- 4. Cluster Characteristics (Heatmap) ---
plt.figure(figsize=(10, 6))
cluster_means = predictions.groupby('settlement_type')[['arrival_score', 'stay_intent_score', 'family_settlement_score']].mean()
sns.heatmap(cluster_means, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("AI Model Logic: How it classifies Pincodes", fontsize=15)
plt.ylabel("Settlement Type", fontsize=12)
plt.xlabel("Activity Score", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(diagrams_path, "model_logic_heatmap.png"))
plt.close()

print(f"All diagrams have been generated and saved in: {diagrams_path}")
