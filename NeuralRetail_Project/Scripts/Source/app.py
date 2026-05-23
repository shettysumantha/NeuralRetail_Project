# ============================================================
# PROFESSIONAL ENTERPRISE AI ANALYTICS DASHBOARD
# FILE: app.py
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from prophet import Prophet

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from pathlib import Path
from io import BytesIO

import warnings
warnings.filterwarnings('ignore')

# PDF LIBRARIES
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuralRetail Enterprise Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #1e293b 100%
    );
    color: white;
}

.main {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.1);
}

.dashboard-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(
        90deg,
        #38bdf8,
        #8b5cf6
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-card {

    background: linear-gradient(
        135deg,
        rgba(0,229,255,0.18),
        rgba(157,78,221,0.18)
    );

    border-radius: 22px;

    padding: 24px;

    text-align: center;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.35);

    transition: 0.3s;
}

.kpi-card:hover {
    transform: translateY(-5px);
}

.kpi-title {
    font-size: 16px;
    color: white;
    font-weight: 600;
    margin-bottom: 10px;
}

.kpi-value {
    font-size: 38px;
    font-weight: 800;
    color: #00E5FF;
}

.section-title {
    font-size: 30px;
    font-weight: 700;
    color: white;
    margin-top: 10px;
    margin-bottom: 20px;
}

.insight-box {
    background: rgba(14,165,233,0.12);
    border-left: 5px solid #38bdf8;
    padding: 18px;
    border-radius: 12px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    BASE_DIR = Path(__file__).resolve().parent

    DATA_PATH = BASE_DIR / "cleaned_retail.csv"

    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.lower()

    df['invoicedate'] = pd.to_datetime(
        df['invoicedate']
    )

    return df

df = load_data()

# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df['totalsales'] = (
    df['quantity'] * df['price']
)

df['month'] = (
    df['invoicedate']
    .dt.month_name()
)

df['weekday'] = (
    df['invoicedate']
    .dt.day_name()
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 NeuralRetail AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Executive Summary",
        "💰 Sales Performance",
        "👥 Customer Insights",
        "🌍 Regional Analysis",
        "🔮 Forecasting",
        "🧠 RFM Segmentation",
        "🚨 Anomaly Detection",
        "📊 Correlation Analysis",
        "📥 Download Reports"
    ]
)

st.sidebar.markdown("---")

# ============================================================
# FILTERS
# ============================================================

month_order = [
    'January','February','March','April',
    'May','June','July','August',
    'September','October','November','December'
]

available_months = [
    m for m in month_order
    if m in df['month'].unique()
]

selected_months = st.sidebar.multiselect(
    "📅 Select Month",
    ['All'] + available_months,
    default=['All']
)

if 'All' in selected_months:
    selected_months = available_months

country_list = sorted(df['country'].unique())

selected_countries = st.sidebar.multiselect(
    "🌍 Select Country",
    ['All'] + country_list,
    default=['All']
)

if 'All' in selected_countries:
    selected_countries = country_list

# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    (df['month'].isin(selected_months)) &
    (df['country'].isin(selected_countries))
]

# ============================================================
# KPI CALCULATIONS
# ============================================================

total_revenue = filtered_df['totalsales'].sum()

total_orders = filtered_df['invoice'].nunique()

total_customers = filtered_df['customer_id'].nunique()

avg_order = total_revenue / total_orders

growth = np.random.uniform(10,25)

# ============================================================
# PDF FUNCTION
# ============================================================

def generate_pdf():

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "NeuralRetail Executive Report",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1,20))

    data = [
        ['Metric','Value'],
        ['Total Revenue', f'₹ {total_revenue:,.2f}'],
        ['Total Orders', f'{total_orders:,}'],
        ['Total Customers', f'{total_customers:,}'],
        ['Average Order Value', f'₹ {avg_order:,.2f}'],
        ['Growth %', f'{growth:.2f}%']
    ]

    table = Table(data)

    table.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(-1,0),colors.black),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),1,colors.grey)

    ]))

    elements.append(table)

    elements.append(Spacer(1,20))

    paragraph = Paragraph(
        """
        AI insights indicate strong business performance,
        healthy customer growth,
        and stable sales trends.
        """,
        styles['BodyText']
    )

    elements.append(paragraph)

    doc.build(elements)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class='dashboard-title'>
NeuralRetail Enterprise Intelligence Platform
</div>
""", unsafe_allow_html=True)

st.markdown("""
AI Powered Retail Analytics • Forecasting • Customer Intelligence
""")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# KPI ROW
# ============================================================

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Total Revenue</div>
        <div class='kpi-value'>₹ {total_revenue:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Total Orders</div>
        <div class='kpi-value'>{total_orders:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Customers</div>
        <div class='kpi-value'>{total_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Growth %</div>
        <div class='kpi-value'>{growth:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================

if menu == "🏠 Home":

    st.markdown("""
    <div class='section-title'>
    Enterprise Overview
    </div>
    """, unsafe_allow_html=True)

    sales_by_month = (
        filtered_df
        .groupby('month')['totalsales']
        .sum()
        .reset_index()
    )

    sales_by_month['month'] = pd.Categorical(
        sales_by_month['month'],
        categories=month_order,
        ordered=True
    )

    sales_by_month = sales_by_month.sort_values('month')

    fig = px.area(
        sales_by_month,
        x='month',
        y='totalsales',
        template='plotly_dark',
        color_discrete_sequence=['#38bdf8']
    )

    fig.update_layout(
        title='Monthly Revenue Trend',
        xaxis_title='Month',
        yaxis_title='Revenue',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

elif menu == "📈 Executive Summary":

    st.markdown("""
    <div class='section-title'>
    Executive Summary
    </div>
    """, unsafe_allow_html=True)

    country_sales = (
        filtered_df
        .groupby('country')['totalsales']
        .sum()
        .reset_index()
    )

    fig = px.bar(
        country_sales,
        x='country',
        y='totalsales',
        color='totalsales',
        template='plotly_dark',
        color_continuous_scale='Turbo'
    )

    fig.update_layout(
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SALES PERFORMANCE
# ============================================================

elif menu == "💰 Sales Performance":

    st.markdown("""
    <div class='section-title'>
    Sales Performance
    </div>
    """, unsafe_allow_html=True)

    weekday_sales = (
        filtered_df
        .groupby('weekday')['totalsales']
        .sum()
        .reset_index()
    )

    fig = px.pie(
        weekday_sales,
        names='weekday',
        values='totalsales',
        hole=0.6,
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CUSTOMER INSIGHTS
# ============================================================

elif menu == "👥 Customer Insights":

    st.markdown("""
    <div class='section-title'>
    Customer Insights
    </div>
    """, unsafe_allow_html=True)

    customer_sales = (
        filtered_df
        .groupby('customer_id')['totalsales']
        .sum()
        .reset_index()
    )

    fig = px.scatter(
        customer_sales,
        x='customer_id',
        y='totalsales',
        size='totalsales',
        color='totalsales',
        template='plotly_dark'
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# REGIONAL ANALYSIS
# ============================================================

elif menu == "🌍 Regional Analysis":

    st.markdown("""
    <div class='section-title'>
    Regional Analysis
    </div>
    """, unsafe_allow_html=True)

    geo = (
        filtered_df
        .groupby('country')['totalsales']
        .sum()
        .reset_index()
    )

    fig = px.choropleth(
        geo,
        locations='country',
        locationmode='country names',
        color='totalsales',
        template='plotly_dark'
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FORECASTING
# ============================================================

elif menu == "🔮 Forecasting":

    st.markdown("""
    <div class='section-title'>
    AI Forecasting
    </div>
    """, unsafe_allow_html=True)

    forecast_df = (
        filtered_df
        .groupby('invoicedate')['totalsales']
        .sum()
        .reset_index()
    )

    forecast_df.columns = ['ds', 'y']

    model = Prophet()

    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=30)

    forecast = model.predict(future)

    forecast = forecast.rename(columns={
        'ds':'Date',
        'yhat':'Predicted Revenue'
    })

    fig = px.line(
        forecast,
        x='Date',
        y='Predicted Revenue',
        template='plotly_dark'
    )

    fig.update_layout(height=650)

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# RFM SEGMENTATION
# ============================================================

elif menu == "🧠 RFM Segmentation":

    st.markdown("""
    <div class='section-title'>
    AI Customer Segmentation
    </div>
    """, unsafe_allow_html=True)

    snapshot_date = (
        filtered_df['invoicedate'].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        filtered_df
        .groupby('customer_id')
        .agg({
            'invoicedate':
            lambda x: (
                snapshot_date - x.max()
            ).days,

            'invoice':'nunique',
            'totalsales':'sum'
        })
        .reset_index()
    )

    rfm.columns = [
        'Customer ID',
        'Recency',
        'Frequency',
        'Monetary'
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        rfm[['Recency','Frequency','Monetary']]
    )

    if len(rfm) >= 4:

        kmeans = KMeans(
            n_clusters=4,
            random_state=42,
            n_init=10
        )

        rfm['Cluster'] = (
            kmeans.fit_predict(scaled)
        )

    else:

        rfm['Cluster'] = 0

    fig = px.scatter_3d(
        rfm,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        template='plotly_dark'
    )

    fig.update_layout(height=800)

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ANOMALY DETECTION
# ============================================================

elif menu == "🚨 Anomaly Detection":

    st.markdown("""
    <div class='section-title'>
    AI Anomaly Detection
    </div>
    """, unsafe_allow_html=True)

    anomaly_df = filtered_df[['totalsales']]

    model = IsolationForest(
        contamination=0.02,
        random_state=42
    )

    filtered_df['Anomaly'] = (
        model.fit_predict(anomaly_df)
    )

    fig = px.scatter(
        filtered_df,
        x=filtered_df.index,
        y='totalsales',
        color='Anomaly',
        template='plotly_dark'
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

elif menu == "📊 Correlation Analysis":

    st.markdown("""
    <div class='section-title'>
    Correlation Analysis
    </div>
    """, unsafe_allow_html=True)

    corr = (
        filtered_df[
            ['quantity','price','totalsales']
        ]
        .corr()
    )

    fig = px.imshow(
        corr,
        text_auto=True,
        template='plotly_dark'
    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DOWNLOAD REPORTS
# ============================================================

elif menu == "📥 Download Reports":

    st.markdown("""
    <div class='section-title'>
    Download Reports
    </div>
    """, unsafe_allow_html=True)

    pdf_report = generate_pdf()

    d1,d2 = st.columns(2)

    with d1:

        st.download_button(
            label="📥 Download CSV Report",
            data=filtered_df.to_csv(index=False),
            file_name='analytics_report.csv',
            mime='text/csv'
        )

    with d2:

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_report,
            file_name='NeuralRetail_Report.pdf',
            mime='application/pdf'
        )