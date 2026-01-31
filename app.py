import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Aadhaar Settlement AI",
    page_icon="🌍",
    layout="centered"
)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('settlement_model.joblib')
    scaler = joblib.load('scaler.joblib')
    cluster_map = joblib.load('cluster_map.joblib')
    lookup_data = pd.read_csv('pincode_lookup.csv')
    return model, scaler, cluster_map, lookup_data

try:
    model, scaler, cluster_map, lookup_data = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Title and Description
st.title("🌍 Invisible Internal Migration & Settlement AI")
st.markdown("""
This AI model predicts population shifts and settlement patterns using Aadhaar data trends.
Enter a **Pincode** to see if it's a growing "Boom Town" or a stable area.
""")

# Input
pincode_input = st.number_input("Enter Pincode", min_value=110001, max_value=999999, step=1, value=411001)

if st.button("Predict Settlement Type"):
    # Check if pincode exists in our lookup data
    result = lookup_data[lookup_data['pincode'] == pincode_input]
    
    if not result.empty:
        row = result.iloc[0]
        st.subheader(f"Results for Pincode: {pincode_input}")
        
        # Display Location Info
        st.write(f"**State:** {row['state']}")
        st.write(f"**District:** {row['district']}")
        
        # Display Classification with styling
        settlement_type = row['settlement_type']
        if "Boom Town" in settlement_type:
            st.success(f"Classification: **{settlement_type}** 🚀")
            st.info("Advice: High infrastructure demand expected. Prioritize schools and hospitals.")
        elif "Stable" in settlement_type:
            st.warning(f"Classification: **{settlement_type}** 🏠")
            st.info("Advice: Steady growth. Maintain existing infrastructure.")
        else:
            st.info(f"Classification: **{settlement_type}** 📍")
            st.info("Advice: Low activity. No immediate infrastructure surge predicted.")
            
        # Show the scores
        st.write("---")
        st.write("**Data Trends in this area:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Arrival Score", int(row['arrival_score']))
        col2.metric("Stay Intent", int(row['stay_intent_score']))
        col3.metric("Family Score", int(row['family_settlement_score']))
        
    else:
        st.error("Pincode not found in our current dataset. This might be a very new or low-activity area.")

# Footer
st.markdown("---")
st.markdown("Developed for the UIDAI Data Hackathon 2026")
