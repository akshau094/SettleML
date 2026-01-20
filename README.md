# Invisible Internal Migration & Settlement AI

This project was developed for the **Online Hackathon on Data-Driven Innovation on Aadhaar 2026**.

## 🚀 Overview
The "Invisible Internal Migration & Settlement AI" is a data-driven solution designed to detect and predict population movement and settlement behavior across India using anonymized Aadhaar datasets.

### The Problem
Traditional census data is updated only once every decade, leading to "lagged" urban planning. This results in infrastructure (schools, hospitals, roads) being built where people *used to be*, rather than where they *are settling today*.

### The Solution
By analyzing temporal correlations across three key datasets—Enrolment, Demographic, and Biometric updates—this AI model identifies:
1. **Boom Towns:** Rapidly growing areas with high new arrivals and permanent stay intent.
2. **Stable Areas:** Regions with steady population density.
3. **Low Activity Areas:** Areas with minimal new settlement.

## 🧠 AI Logic: The "Settlement Signature"
The model identifies a 3-step sequence:
1. **Arrival (Enrolment Data):** New Aadhaar registrations in a Pincode.
2. **Intent (Demographic Data):** Address updates signaling a plan to stay.
3. **Stability (Biometric Data):** Updates for children (ages 5-17), signaling family settlement and long-term infrastructure needs (schools).

## 🛠️ Project Structure
- `preprocess.py`: Fuses and cleans the raw biometric, demographic, and enrolment datasets.
- `visualize.py`: Performs exploratory data analysis (EDA) to find top growing districts.
- `train_model.py`: Uses K-Means Clustering to classify over 33,000 Pincodes.
- `generate_diagrams.py`: Generates visual charts and interactive heatmaps for the presentation.
- `diagrams/`: Contains the visual output of the analysis.

## 📊 Impact
This tool enables real-time urban planning, allowing government authorities to anticipate housing demand, school placement, and smart city infrastructure needs before overcrowding occurs.

---
Developed for UIDAI Data Hackathon 2026.
