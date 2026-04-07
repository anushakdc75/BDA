%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI
import os
import time

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Dynamic Pricing Dashboard", layout="wide")

st.title("AI Dynamic Pricing Simulator")

# ---------------- API KEY ----------------

# ---------------- SIDEBAR ----------------
demand = st.sidebar.slider("Demand", 1, 100, 60)
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Storm"])
time_val = st.sidebar.selectbox("Time", ["Off-Peak", "Normal", "Peak"])

traffic_map = {"Low":1, "Medium":2, "High":3}
weather_map = {"Clear":1, "Rain":2, "Storm":3}
time_map = {"Off-Peak":1, "Normal":2, "Peak":3}

# ---------------- DATA ----------------
np.random.seed(42)

df = pd.DataFrame({
    "demand": np.random.randint(1,100,500),
    "traffic": np.random.randint(1,4,500),
    "weather": np.random.randint(1,4,500),
    "time": np.random.randint(1,4,500),
})

df["price"] = df["demand"]*2 + df["traffic"]*20 + df["weather"]*15 + df["time"]*25

# ---------------- BIG DATA ----------------
st.subheader("Big Data Simulation")

big_df = pd.DataFrame({
    "demand": np.random.randint(1,100,100000),
    "traffic": np.random.randint(1,4,100000),
    "weather": np.random.randint(1,4,100000),
    "time": np.random.randint(1,4,100000),
})
st.write("Dataset size:", len(big_df))

# ---------------- STREAMING ----------------
if st.button("Simulate Live Data"):
    for i in range(3):
        st.write({
            "demand": np.random.randint(1,100),
            "traffic": np.random.randint(1,4),
            "weather": np.random.randint(1,4),
            "time": np.random.randint(1,4)
        })
        time.sleep(1)

# ---------------- MODEL ----------------
X = df[["demand","traffic","weather","time"]]
y = df["price"]

model = RandomForestRegressor()
model.fit(X,y)

# ---------------- PREDICTION ----------------
input_data = np.array([[demand,
                        traffic_map[traffic],
                        weather_map[weather],
                        time_map[time_val]]])

price = model.predict(input_data)[0]

st.metric("Predicted Price", f"₹{price:.2f}")

# ================= VISUALIZATIONS =================

# 1
st.subheader("1. Demand vs Price Trend")
fig = px.line(df.sort_values("demand"), x="demand", y="price")
st.plotly_chart(fig)
st.caption("X = Demand, Y = Price")

# 2
st.subheader("2. Traffic Impact")
fig = px.bar(df, x="traffic", y="price", color="traffic")
st.plotly_chart(fig)
st.caption("X = Traffic, Y = Price")

# 3
st.subheader("3. Weather Distribution")
fig = px.box(df, x="weather", y="price")
st.plotly_chart(fig)
st.caption("X = Weather, Y = Price")

# 4
st.subheader("4. Time Distribution")
fig = px.violin(df, x="time", y="price")
st.plotly_chart(fig)
st.caption("X = Time, Y = Price")

# 5
st.subheader("5. Heatmap")
pivot = df.pivot_table(values="price", index="traffic", columns="time")
fig = px.imshow(pivot)
st.plotly_chart(fig)
st.caption("X = Time, Y = Traffic")

# 6
st.subheader("6. 3D Relationship")
fig = px.scatter_3d(df, x="demand", y="traffic", z="price")
st.plotly_chart(fig)

# 7
st.subheader("7. Feature Importance")
importance = model.feature_importances_
features = ["demand","traffic","weather","time"]
imp_df = pd.DataFrame({"feature":features,"importance":importance})
fig = px.bar(imp_df, x="feature", y="importance")
st.plotly_chart(fig)

# 8
st.subheader("8. Price Distribution")
fig = px.histogram(df, x="price")
st.plotly_chart(fig)

# 9
st.subheader("9. Scatter Matrix")
fig = px.scatter_matrix(df)
st.plotly_chart(fig)

# 10
st.subheader("10. Bubble Chart")
fig = px.scatter(df, x="demand", y="price",
                 size="weather", color="traffic")
st.plotly_chart(fig)

# 11
st.subheader("11. Animated Simulation")
df["frame"] = np.random.randint(1,10,len(df))
fig = px.scatter(df, x="demand", y="price",
                 animation_frame="frame",
                 color="traffic")
st.plotly_chart(fig)

# 12
st.subheader("12. Correlation Heatmap")
corr = df.corr()
fig = px.imshow(corr)
st.plotly_chart(fig)

# ---------------- MAP ----------------
st.subheader("Ride Demand Map")

map_df = pd.DataFrame({
    "lat": np.random.uniform(12.90, 13.10, 200),
    "lon": np.random.uniform(77.50, 77.70, 200),
})
st.map(map_df)

# ---------------- LLM ----------------
st.subheader("AI Explanation")

if st.button("Explain Price"):
    prompt = f"""
    The ride price is {price:.2f}.
    Demand is {demand}, traffic is {traffic}, weather is {weather}, and time is {time_val}.
    Explain clearly why this price is high or low.
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    st.write(res.choices[0].message.content)

