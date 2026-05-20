import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# LOAD DATASETS
# ============================================================

rfm = pd.read_csv(
    'final_customer_data.csv'
)

daily_sales = pd.read_csv(
    'final_sales_data.csv'
)

forecast = pd.read_csv(
    'sales_forecast.csv'
)

# ============================================================
# DASHBOARD TITLE
# ============================================================

st.title(
    'NeuralRetail Analytics Dashboard'
)

st.markdown(
    'Retail Forecasting & Customer Intelligence'
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header('Dashboard Filters')

selected_cluster = st.sidebar.selectbox(
    'Select Customer Cluster',
    sorted(rfm['Cluster'].unique())
)

# ============================================================
# KPI CARDS
# ============================================================

total_revenue = daily_sales[
    'TotalSales'
].sum()

avg_sales = daily_sales[
    'TotalSales'
].mean()

total_customers = rfm.shape[0]

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        'Total Revenue',
        f'₹ {total_revenue:,.2f}'
    )

with col2:

    st.metric(
        'Average Daily Sales',
        f'₹ {avg_sales:,.2f}'
    )

with col3:

    st.metric(
        'Total Customers',
        total_customers
    )

# ============================================================
# SALES TREND
# ============================================================

fig_sales = px.line(
    daily_sales,
    x='invoicedate',
    y='TotalSales',
    title='Daily Sales Trend'
)

st.plotly_chart(fig_sales)

# ============================================================
# FORECAST VISUALIZATION
# ============================================================

fig_forecast = px.line(
    forecast,
    x='ds',
    y='yhat',
    title='Sales Forecast'
)

st.plotly_chart(fig_forecast)

# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

cluster_data = (
    rfm.groupby('Cluster')[
        'Monetary'
    ]
    .mean()
    .reset_index()
)

fig_cluster = px.bar(
    cluster_data,
    x='Cluster',
    y='Monetary',
    title='Average Spending by Cluster'
)

st.plotly_chart(fig_cluster)

# ============================================================
# FILTERED CUSTOMER VIEW
# ============================================================

filtered_customers = rfm[
    rfm['Cluster'] == selected_cluster
]

st.subheader(
    'Filtered Customers'
)

st.dataframe(
    filtered_customers.head(20)
)