import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
@st.cache_data
def load_data():
    data_path = "../data/merged_train_data.parquet"  # Update this path if necessary
    data = pd.read_parquet(data_path, engine='pyarrow')
    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.day_name()  # Add day names for plotting
    return data

data = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Multi-select for stores
unique_stores = data['Store'].unique()
selected_stores = st.sidebar.multiselect(
    "Select Stores:", 
    options=unique_stores, 
    default=unique_stores[:5]
)

# Sales threshold filter
min_sales, max_sales = st.sidebar.slider(
    "Select Sales Range:", 
    min_value=int(data['Sales'].min()), 
    max_value=int(data['Sales'].max()), 
    value=(int(data['Sales'].min()), int(data['Sales'].max()))
)

# Promotion filter
promo_filter = st.sidebar.selectbox(
    "Promotion Status:", 
    options=["All", "Promotion", "No Promotion"]
)

# Apply filters
filtered_data = data[(data['Store'].isin(selected_stores)) &
                     (data['Sales'] >= min_sales) &
                     (data['Sales'] <= max_sales)]

if promo_filter == "Promotion":
    filtered_data = filtered_data[filtered_data['Promo'] == 1]
elif promo_filter == "No Promotion":
    filtered_data = filtered_data[filtered_data['Promo'] == 0]

# Customizable Themes
st.sidebar.header("Customization")
color_palettes = ["viridis", "magma", "coolwarm", "Set2", "Spectral"]
selected_palette = st.sidebar.selectbox(
    "Select Chart Color Palette:", 
    options=color_palettes, 
    index=0
)

sns.set_palette(selected_palette)

# 1. Sales Trends Over Time
def plot_sales_trends(data):
    st.subheader("Sales Trends Over Time")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Date', y='Sales', marker='o')
    plt.title("Sales Trends", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

# 2. Promotion vs. Non-Promotion Sales Comparison
def plot_promotion_comparison(data):
    st.subheader("Promotion vs. Non-Promotion Sales")
    promo_comparison = data.groupby('Promo')['Sales'].mean().reset_index()
    promo_comparison['Promo'] = promo_comparison['Promo'].map({1: "Promotion", 0: "No Promotion"})
    plt.figure(figsize=(8, 5))
    sns.barplot(data=promo_comparison, x='Promo', y='Sales')
    plt.title("Average Sales: Promotion vs. Non-Promotion", fontsize=14)
    plt.xlabel("Promotion Status", fontsize=12)
    plt.ylabel("Average Sales", fontsize=12)
    st.pyplot(plt.gcf())

# 3. Sales Distribution by Day of the Week
def plot_sales_by_day(data):
    st.subheader("Sales Distribution by Day of the Week")
    day_sales = data.groupby('DayOfWeek')['Sales'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=day_sales, 
        x='DayOfWeek', 
        y='Sales', 
        order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    plt.title("Average Sales by Day of the Week", fontsize=14)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Average Sales", fontsize=12)
    st.pyplot(plt.gcf())

# Generate Charts
plot_sales_trends(filtered_data)
plot_promotion_comparison(filtered_data)
plot_sales_by_day(filtered_data)
