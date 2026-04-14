# Dynamic Pricing Engine for Ride-Sharing using Big Data Analytics

## Overview

This project implements a Dynamic Pricing Engine for ride-sharing platforms using machine learning and big data simulation. The system dynamically adjusts ride prices based on key factors such as demand, traffic conditions, weather, and time of day.

An interactive dashboard is built using Streamlit, and Explainable AI is integrated using OpenAI API to provide clear explanations for pricing decisions.

---

## Features

- Machine Learning based price prediction (Random Forest)
- Interactive Streamlit dashboard
- Big data simulation (100,000+ records)
- Real-time input controls (demand, traffic, weather, time)
- Multiple data visualizations using Plotly
- Explainable AI using OpenAI API
- Live data simulation

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (RandomForestRegressor)
- Streamlit
- Plotly
- OpenAI API

---

## Workflow

1. Synthetic Data Generation  
2. Data Preprocessing  
3. Big Data Simulation  
4. Model Training (Random Forest)  
5. Price Prediction  
6. Streamlit Dashboard  
7. Data Visualization  
8. Explainable AI (OpenAI API)  
9. Final Output (Price + Explanation)

---

## Dataset

The dataset is synthetically generated and contains:

- Demand (1–100)
- Traffic (Low, Medium, High → 1–3)
- Weather (Clear, Rain, Storm → 1–3)
- Time (Off-Peak, Normal, Peak → 1–3)

### Price Formula

Price = Demand × 2 + Traffic × 20 + Weather × 15 + Time × 25

---

## Machine Learning Model

- Model: Random Forest Regressor  
- Inputs: Demand, Traffic, Weather, Time  
- Output: Predicted Price  

The model learns patterns from synthetic data and predicts dynamic pricing.

---

## Big Data Simulation

A dataset of 100,000 records is generated to simulate large-scale data processing and test system performance.

---

## Dashboard

The Streamlit dashboard allows users to:

- Adjust demand (slider)
- Select traffic level
- Select weather condition
- Select time category
- View predicted price instantly

---

## Visualizations

- Demand vs Price Trend  
- Traffic Impact  
- Weather Distribution  
- Time Distribution  
- Heatmap  
- 3D Scatter Plot  
- Feature Importance  
- Correlation Matrix  

---

## Explainable AI

OpenAI API is used to generate explanations for predicted prices.

Example:

"The ride price is high due to high demand, peak time, and increased traffic."

This improves transparency and user understanding.

---

## Installation

Clone the repository:

git clone https://github.com/anushakdc75/BDA.git  
cd BDA  

Install dependencies:

pip install -r requirements.txt  

Run the app:

streamlit run app.py  

---

## Usage

- Open the dashboard in browser  
- Adjust input parameters  
- View predicted price  
- Click "Explain Price" for AI explanation  
- Explore visualizations  

---

## Results

- Demand has the highest impact on price  
- Peak time significantly increases price  
- Traffic and weather moderately affect pricing  
- Model successfully simulates real-world dynamic pricing  

---

## Future Enhancements

- Real-time streaming using Kafka  
- Cloud deployment  
- Integration with maps API  
- Deep learning models  
- Personalized pricing  

---

## Author

Anusha K  
B.Tech CSE (Data Analytics)  
Alliance University
