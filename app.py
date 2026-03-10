import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Restaurant AI Dashboard", layout="wide")

st.title("🍽 Restaurant Rating Prediction Dashboard")
st.write("AI powered restaurant rating prediction and analytics")

# Load model
model = joblib.load("restaurant_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")
cuisine_encoder = joblib.load("cuisine_encoder.pkl")

# Load dataset
df = pd.read_csv("Dataset.csv")

# Sidebar
st.sidebar.header("Restaurant Details")

city = st.sidebar.selectbox("City", city_encoder.classes_)
cuisine = st.sidebar.selectbox("Cuisine", cuisine_encoder.classes_)
price_range = st.sidebar.slider("Price Range", 1, 4, 2)
votes = st.sidebar.slider("Votes", 0, 5000, 100)
table_booking = st.sidebar.selectbox("Table Booking", ["Yes", "No"])
online_delivery = st.sidebar.selectbox("Online Delivery", ["Yes", "No"])
average_cost = st.sidebar.slider("Average Cost for Two", 0, 5000, 500)

# Encoding
city_encoded = city_encoder.transform([city])[0]
cuisine_encoded = cuisine_encoder.transform([cuisine])[0]
table_booking = 1 if table_booking == "Yes" else 0
online_delivery = 1 if online_delivery == "Yes" else 0

# Prediction
if st.sidebar.button("Predict Rating"):

    input_data = pd.DataFrame({
        "City": [city_encoded],
        "Cuisines": [cuisine_encoded],
        "Average Cost for two": [average_cost],
        "Price range": [price_range],
        "Votes": [votes],
        "Has Table booking": [table_booking],
        "Has Online delivery": [online_delivery]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("⭐ Predicted Rating")
    st.success(round(prediction, 2))


# Dashboard analytics
st.header("📊 Restaurant Data Analytics")

col1, col2 = st.columns(2)

# Rating distribution
with col1:
    fig = px.histogram(df, x="Aggregate rating", nbins=20, title="Rating Distribution")
    st.plotly_chart(fig)

# Price range vs rating
with col2:
    fig2 = px.box(df, x="Price range", y="Aggregate rating", title="Price Range vs Rating")
    st.plotly_chart(fig2)


# Top Cities
st.subheader("🏙 Top Restaurant Cities")

top_cities = df["City"].value_counts().head(10).reset_index()
top_cities.columns = ["City","Restaurants"]

fig3 = px.bar(top_cities, x="City", y="Restaurants", title="Top Cities with Restaurants")
st.plotly_chart(fig3)


# Top cuisines
st.subheader("🍜 Popular Cuisines")

top_cuisines = df["Cuisines"].value_counts().head(10).reset_index()
top_cuisines.columns = ["Cuisine","Count"]

fig4 = px.bar(top_cuisines, x="Cuisine", y="Count", title="Top Cuisines")
st.plotly_chart(fig4)


# Map visualization
if "Latitude" in df.columns and "Longitude" in df.columns:

    st.subheader("🌍 Restaurant Locations")

    map_data = df[["Latitude","Longitude","Restaurant Name"]].dropna().head(500)

    fig5 = px.scatter_mapbox(
        map_data,
        lat="Latitude",
        lon="Longitude",
        hover_name="Restaurant Name",
        zoom=3
    )

    fig5.update_layout(mapbox_style="open-street-map")

    st.plotly_chart(fig5)