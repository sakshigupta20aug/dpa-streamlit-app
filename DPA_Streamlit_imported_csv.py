# streamlit_app.py
# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import pyodbc
import os
from hashlib import sha256  # 🔒 For secure login

# --------------------------
# 🔐 USER AUTHENTICATION
# --------------------------
USER_CREDENTIALS = {
    "kavita": sha256("kavita123".encode()).hexdigest(),
    "sakshi": sha256("sakshi123".encode()).hexdigest(),
    "amita": sha256("amita123".encode()).hexdigest(),
    "shreya": sha256("shreya123".encode()).hexdigest()
}

def login():
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in USER_CREDENTIALS:
            if sha256(password.encode()).hexdigest() == USER_CREDENTIALS[username]:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success(f"Welcome, {username.title()}! 🎉")
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
        else:
            st.error("❌ Username not found.")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

if not st.session_state.logged_in:
    login()
    st.stop()


# --------------------------
# 🔧 PAGE SETUP
# --------------------------
st.set_page_config(page_title="E-commerce Dashboard", layout="wide")
sns.set(style="whitegrid")

# --------------------------
# 🎛️ SIDEBAR NAVIGATION
# --------------------------
# Sidebar
st.sidebar.title("Navigation")

# Show logged-in user
if st.session_state.get("user"):
    st.sidebar.markdown(f"👤 Logged in as: **{st.session_state.user.title()}**")

# Navigation options
section = st.sidebar.radio("Go to", [
    "Executive KPI Dashboard",  # renamed from Overview
    "Website Analytics",
    "Product Analytics",
    "Investor Analytics",
    "Marketing Analytics",
    "Customer Insights",
    "Behavioral Segmentation",
    "Campaign Performance",
    "Cohort Analysis",
    "RFM Segmentation",
    "Session Funnel",
    "Conversion Funnel"
])



# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


if "logged_in" in st.session_state and not st.session_state.logged_in:
    st.success("Successfully logged out.")

# --------------------------
import os  # Make sure this is at the top of your file

# 📁 CSV FILE LOADING FROM GITHUB-CLONED RELATIVE PATH
import os

def load_data():
    try:
        data_path = os.path.join(os.getcwd(), "data")  # ✅ Relative path for Render
        orders = pd.read_csv(os.path.join(data_path, "orders.csv"))
        order_items = pd.read_csv(os.path.join(data_path, "order_items.csv"))
        order_item_refunds = pd.read_csv(os.path.join(data_path, "order_item_refunds.csv"))
        products = pd.read_csv(os.path.join(data_path, "products.csv"))
        website_pageviews = pd.read_csv(os.path.join(data_path, "website_pageviews.csv"))
        website_sessions = pd.read_csv(os.path.join(data_path, "website_sessions.csv"))
        st.success("✅ CSV files loaded successfully")
        return [orders, order_items, order_item_refunds, products, website_pageviews, website_sessions]
    except Exception as e:
        st.error(f"❌ Failed to load CSV files: {e}")
        return [None]*6

orders, order_items, order_item_refunds, products, website_pageviews, website_sessions = load_data()

if any(df is None or df.empty for df in [orders, order_items, order_item_refunds, products, website_pageviews, website_sessions]):
    st.error("❌ One or more required CSV files are missing or empty.")
    st.stop()



# --------------------------
# 🧹 PREPROCESSING
# --------------------------
df_filtered = website_sessions.copy()
df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'])
df_filtered['year'] = df_filtered['created_at'].dt.year
df_filtered['quarter'] = df_filtered['created_at'].dt.quarter

# --------------------------
# 📊 KPI CALCULATIONS (before Overview)
# --------------------------

# Revenue & Cost
Total_Revenue = orders['price_usd'].sum()
total_cost = orders['cogs_usd'].sum()

# ✅ Safe Refund Calculation using order_id join (if available)
if 'order_id' in order_items.columns:
    refunds_joined = order_item_refunds.merge(order_items[['order_item_id', 'order_id']], on='order_item_id', how='left')
    if 'order_id' in refunds_joined.columns:
        refunded_order_ids = refunds_joined['order_id'].dropna().unique()
        Total_Refund = orders[orders['order_id'].isin(refunded_order_ids)]['price_usd'].sum()
    else:
        Total_Refund = 0
else:
    Total_Refund = 0

# 💰 Final Profit (Revenue - Refund - Cost)
profit = Total_Revenue - Total_Refund - total_cost




# Other Metrics
total_orders = len(orders)
total_sessions = len(website_sessions)
Total_buyers = orders['user_id'].nunique()


# Conversion Rate & Bounce
Conversion_Rate = (orders['website_session_id'].nunique() / website_sessions['website_session_id'].nunique()) * 100
# bounce_rate = (website_sessions['is_bounced'].sum() / len(website_sessions)) * 100

# Buyer metrics
buyers = orders.groupby('user_id').size()
one_time_buyers = (buyers == 1).sum()
returning_buyers = (buyers > 1).sum()
pct_returning_buyers = returning_buyers / buyers.shape[0] * 100
pct_one_time_buyers = one_time_buyers / buyers.shape[0] * 100
Avg_revenue_per_buyer = Total_Revenue / buyers.shape[0]
Avg_revenue_per_order = Total_Revenue / total_orders
avg_profit_per_buyer = profit / buyers.shape[0]

# Items per order
items_per_order = order_items.groupby('order_id')['order_item_id'].count()
avg_item_per_order = items_per_order.mean()

# User behavior
user_sessions = website_sessions.groupby('user_id')['website_session_id'].count()
one_time_users = (user_sessions == 1).sum()
returning_users = (user_sessions > 1).sum()
pct_returning_users = returning_users / user_sessions.shape[0] * 100
pct_one_time_users = one_time_users / user_sessions.shape[0] * 100
avg_sessions_per_user = user_sessions.mean()

# Refund metrics
Item_refunded_count = len(order_item_refunds)
total_items_ordered = len(order_items)
pct_returned_items = (Item_refunded_count / total_items_ordered) * 100
 # Total_Refund = refunded_amount

# --------------------------
# 📊 KPI DASHBOARD (OVERVIEW)
# --------------------------
if section == "Executive KPI Dashboard":
    st.markdown("""
    <h2 style='color:#333'>📊 Executive KPI Dashboard</h2>
    <p style='font-size:16px;'>Track revenue, customer behavior, and performance insights across the funnel.</p>
    """, unsafe_allow_html=True)

    def custom_kpi(label, value):
        st.markdown(f"<div style='background-color:#f0f2f6;padding:12px 16px;border-radius:12px;text-align:center;margin-bottom:10px;'>"
                    f"<h5 style='color:#666'>{label}</h5>"
                    f"<h2 style='color:#0072C6;margin:0'>{value}</h2></div>", unsafe_allow_html=True)

    # Core Metrics
    st.subheader("🔢 Core Metrics")
    col1, col2, col3 = st.columns(3)
    with col1: custom_kpi("🛒 Orders", f"{total_orders:,}")
    with col2: custom_kpi("🌍 Sessions", f"{total_sessions/1_000_000:,.2f}M")
    with col3: custom_kpi("👤 Buyers", f"{Total_buyers:,}")

    col4, col5, col6 = st.columns(3)
    with col4: custom_kpi("💵 Revenue", f"${Total_Revenue / 1_000_000:,.2f}M")
    with col5: custom_kpi("📉 COGS", f"${total_cost / 1_000_000:,.2f}M")
    with col6: custom_kpi("💰 Profit", f"${profit / 1_000_000:,.2f}M")

    col7, col8, col9 = st.columns(3)
    with col7: custom_kpi("📈 Conversion", f"{Conversion_Rate:.2f}%")
    with col8: custom_kpi("📦 Total Orders", f"{total_orders:,}")
    # with col9: custom_kpi("🧾 Net Revenue", f"${Net_Revenue / 1_000_000:,.2f}M")

    # Buyer Metrics
    st.subheader("🧑‍💼 Buyer Insights")
    col10, col11, col12 = st.columns(3)
    with col10: custom_kpi("💳 Revenue/Buyer", f"${Avg_revenue_per_buyer:,.2f}")
    with col11: custom_kpi("📦 Revenue/Order", f"${Avg_revenue_per_order:,.2f}")
    with col12: custom_kpi("💹 Profit/Buyer", f"${avg_profit_per_buyer:,.2f}")

    col13, col14, col15 = st.columns(3)
    with col13: custom_kpi("🧾 One-time Buyers", f"{one_time_buyers:,}")
    with col14: custom_kpi("🔁 Repeat Buyers", f"{returning_buyers:,}")
    with col15: custom_kpi("% Repeat", f"{pct_returning_buyers:.2f}%")

    # User Metrics
    st.subheader("🧠 User Engagement")
    col16, col17, col18 = st.columns(3)
    with col16: custom_kpi("👤 One-time Users", f"{one_time_users/1_000_000:,.2f}M")
    with col17: custom_kpi("👥 Returning Users", f"{returning_users/1_000_000:,.2f}M")
    with col18: custom_kpi("📊 Avg Sessions/User", f"{avg_sessions_per_user:.2f}")

    col19, col20 = st.columns(2)
    with col19: custom_kpi("↩️ % Returning Users", f"{pct_returning_users:.2f}%")
    with col20: custom_kpi("🚪 % One-time Users", f"{pct_one_time_users:.2f}%")

    # Refund Metrics
    st.subheader("↩️ Refund Metrics")
    col21, col22, col23 = st.columns(3)
    with col21: custom_kpi("💸 Refunds", f"${Total_Refund / 1_000_000:,.2f}M")
    with col22: custom_kpi("📦 Items Refunded", f"{Item_refunded_count:,}")
    with col23: custom_kpi("📉 % Returned Items", f"{pct_returned_items:.2f}%")


elif section == "Website Analytics":
    st.title("\U0001F310 Website Analytics")
    top_pages = website_pageviews['pageview_url'].value_counts().reset_index()
    top_pages.columns = ['pageview_url', 'views']
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=top_pages.head(10), x='pageview_url', y='views', palette='cool', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    bounce_sessions = website_pageviews.groupby('website_session_id').count()
    bounce_sessions = bounce_sessions[bounce_sessions['pageview_url'] == 1]
    bounce_rate = len(bounce_sessions) / website_sessions.shape[0] * 100
    st.metric("Bounce Rate", f"{bounce_rate:.2f}%")

    thank_you_sessions = website_pageviews[website_pageviews['pageview_url'] == '/thank-you-for-your-order']['website_session_id'].nunique()
    total_sessions = website_sessions['website_session_id'].nunique()
    conversion_rate = thank_you_sessions / total_sessions * 100
    st.metric("Conversion Rate", f"{conversion_rate:.2f}%")

elif section == "Product Analytics":
    st.title("\U0001F4E6 Product Analytics")

    # ✅ Merge to get product names with order items
    order_items_merged = pd.merge(order_items, products, on='product_id', how='left')

    # Group by product name
    product_sales = order_items_merged.groupby('product_name').agg(
        units_sold=('order_item_id', 'count'),
        revenue=('price_usd', 'sum'),
        cogs=('cogs_usd', 'sum')
    ).reset_index()

    product_sales['profit'] = product_sales['revenue'] - product_sales['cogs']

    st.dataframe(product_sales.sort_values(by='revenue', ascending=False))



elif section == "Investor Analytics":
    st.title("📈 Investor Analytics")

    # Basic metrics
    total_orders = orders['order_id'].nunique()
    total_revenue = orders['price_usd'].sum()
    total_cogs = orders['cogs_usd'].sum()
    gross_profit = total_revenue - total_cogs

    # Refunds
    refunded_orders = order_item_refunds[['order_id', 'refund_amount_usd']].copy()
    refunded_orders = refunded_orders.groupby('order_id')['refund_amount_usd'].sum().reset_index()

    # Ensure 'order_id' exists in both DataFrames
    investor_df = orders.merge(refunded_orders, on='order_id', how='left')
    investor_df['refund_amount_usd'] = investor_df['refund_amount_usd'].fillna(0)

    investor_df['net_revenue'] = investor_df['price_usd'] - investor_df['refund_amount_usd']
    investor_df['net_profit'] = investor_df['net_revenue'] - investor_df['cogs_usd']

    net_revenue = investor_df['net_revenue'].sum()
    net_profit = investor_df['net_profit'].sum()
    profit_margin = (net_profit / net_revenue * 100) if net_revenue != 0 else 0

    st.metric("Total Orders", f"{total_orders:,}")
    st.metric("Gross Revenue (USD)", f"${total_revenue:,.2f}")
    st.metric("Net Revenue (USD)", f"${net_revenue:,.2f}")
    st.metric("Net Profit (USD)", f"${net_profit:,.2f}")
    st.metric("Profit Margin", f"{profit_margin:.2f}%")

    # Optional: Show investor_df table
    with st.expander("Show Order-Level Financial Data"):
        st.dataframe(investor_df)


elif section == "Marketing Analytics":
    st.title("\U0001F4F1 Marketing Analytics")
    campaign_perf = website_sessions.groupby(['utm_source', 'utm_campaign']).agg(
        sessions=('website_session_id', 'count')
    ).reset_index()
    thank_you_sessions = website_pageviews[website_pageviews['pageview_url'] == '/thank-you-for-your-order']['website_session_id'].nunique()
    campaign_perf['conversions'] = thank_you_sessions // campaign_perf.shape[0]
    campaign_perf['conversion_rate'] = campaign_perf['conversions'] / campaign_perf['sessions'] * 100
    st.dataframe(campaign_perf)

elif section == "Customer Insights":
    st.title("\U0001F465 Customer Insights")
    orders_with_items = pd.merge(order_items, orders, on='order_id')
    customer_summary = orders_with_items.groupby('user_id').agg(
        total_spent=('price_usd_x', 'sum'),
        total_orders=('order_id', 'nunique')
    ).reset_index()
    refunds_with_orders = pd.merge(order_item_refunds, orders_with_items, on='order_item_id')
    refund_summary = refunds_with_orders.groupby('user_id').agg(
        total_refunds=('refund_amount_usd', 'sum')
    ).reset_index()
    customer_summary = pd.merge(customer_summary, refund_summary, on='user_id', how='left').fillna(0)
    customer_summary['net_spent'] = customer_summary['total_spent'] - customer_summary['total_refunds']
    st.dataframe(customer_summary.head())

elif section == "Behavioral Segmentation":
    st.title("\U0001F4CA Behavioral Segmentation")
    page_counts = website_pageviews.groupby('website_session_id')['pageview_url'].nunique().reset_index(name='unique_pages_viewed')

    def segment_behavior(p):
        if p >= 5:
            return 'Deep Engagement'
        elif p >= 3:
            return 'Moderate Engagement'
        else:
            return 'Low Engagement'

    page_counts['Segment'] = page_counts['unique_pages_viewed'].apply(segment_behavior)
    behavior_counts = page_counts['Segment'].value_counts().reset_index()
    behavior_counts.columns = ['Segment', 'Count']
    fig, ax = plt.subplots()
    sns.barplot(data=behavior_counts, x='Segment', y='Count', palette='Set3', ax=ax)
    st.pyplot(fig)

elif section == "Campaign Performance":
    st.title("\U0001F4CB Campaign Performance")
    
    # Clean and prep
    website_pageviews.columns = website_pageviews.columns.str.strip().str.lower()
    website_sessions.columns = website_sessions.columns.str.strip().str.lower()

    # Session time tracking
    session_times = website_sessions.copy()
    session_times['created_at'] = pd.to_datetime(session_times['created_at'])
    session_times = session_times.sort_values(by=['user_id', 'created_at'])

    session_times['next_created_at'] = session_times.groupby('user_id')['created_at'].shift(-1)
    session_times['session_duration_min'] = (
        (session_times['next_created_at'] - session_times['created_at']).dt.total_seconds() / 60
    )

    # Conversion tagging
    thank_you_sessions = website_pageviews[
        website_pageviews['pageview_url'] == '/thank-you-for-your-order'
    ]['website_session_id'].unique()

    session_times['is_converted'] = session_times['website_session_id'].isin(thank_you_sessions)

    # Drop sessions with unknown duration
    session_times = session_times.dropna(subset=['session_duration_min'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=session_times, x='is_converted', y='session_duration_min', palette='coolwarm', ax=ax)
    plt.xlabel('Converted')
    plt.ylabel('Session Duration (min)')
    plt.xticks([0, 1], ['No', 'Yes'])
    st.pyplot(fig)


elif section == "Cohort Analysis":
    st.title("\U0001F5FA Cohort Analysis")
    sessions = website_sessions.copy()
    sessions['cohort_group'] = sessions.groupby('user_id')['created_at'].transform('min').dt.to_period('M')
    sessions['order_month'] = sessions['created_at'].dt.to_period('M')
    sessions['cohort_index'] = (sessions['order_month'].dt.year - sessions['cohort_group'].dt.year) * 12 + (sessions['order_month'].dt.month - sessions['cohort_group'].dt.month) + 1
    cohort_data = sessions.groupby(['cohort_group', 'cohort_index'])['user_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot_table(index='cohort_group', columns='cohort_index', values='user_id')
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data=retention, annot=True, fmt='.0%', cmap='Blues', ax=ax)
    st.pyplot(fig)

elif section == "RFM Segmentation":
    st.title("\U0001F4C9 RFM Segmentation")
    orders_merged = order_items.copy()
    orders_merged = orders_merged.merge(orders[['order_id', 'website_session_id']], on='order_id', how='left')
    orders_merged = orders_merged.merge(website_sessions[['website_session_id', 'user_id']], on='website_session_id', how='left')
    orders_merged['created_at'] = pd.to_datetime(orders_merged['created_at'], errors='coerce')
    rfm_data = orders_merged.dropna(subset=['user_id', 'created_at'])
    reference_date = rfm_data['created_at'].max()
    rfm = rfm_data.groupby('user_id').agg({
        'created_at': lambda x: (reference_date - x.max()).days,
        'order_id': 'nunique',
        'price_usd': 'sum'
    }).reset_index()
    rfm.columns = ['user_id', 'Recency', 'Frequency', 'Monetary']
    rfm[['R_Score', 'F_Score', 'M_Score']] = pd.DataFrame({
        'R_Score': pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]).astype(int),
        'F_Score': pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int),
        'M_Score': pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    })
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

    def segment_customer(row):
        if row['RFM_Score'] >= 10:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3:
            return 'Loyal Customers'
        elif row['R_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Big Spenders'
        elif row['R_Score'] == 4:
            return 'New Customers'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'At Risk'
        else:
            return 'Others'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Segment', y='Count', data=segment_counts, palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif section == "Conversion Funnel":
    st.title("🛒 Conversion Funnel")

    # Define funnel steps
    funnel_steps = [
        "/home",
        "/category-page",
        "/product-page",
        "/cart",
        "/checkout",
        "/thank-you-for-your-order"
    ]

    # Create a DataFrame to hold funnel counts
    funnel_data = []

    for step in funnel_steps:
        count = website_pageviews[website_pageviews['pageview_url'] == step]['website_session_id'].nunique()
        funnel_data.append({"Step": step, "Sessions": count})

    funnel_df = pd.DataFrame(funnel_data)

    # Visualize Funnel
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=funnel_df, x="Sessions", y="Step", palette="Blues_d", ax=ax)
    plt.xlabel("Unique Sessions")
    plt.ylabel("Funnel Step")
    st.pyplot(fig)

    # Show table
    st.dataframe(funnel_df)


elif section == "Session Funnel":
    st.title("📊 Session Funnel")

    # Prepare session_times if not already defined
    session_times = website_sessions.copy()
    session_times['created_at'] = pd.to_datetime(session_times['created_at'])
    session_times = session_times.sort_values(by=['user_id', 'created_at'])

    # Calculate time until next session
    session_times['next_created_at'] = session_times.groupby('user_id')['created_at'].shift(-1)
    session_times['session_duration_min'] = (session_times['next_created_at'] - session_times['created_at']).dt.total_seconds() / 60

    # Identify sessions that converted
    thank_you_sessions = website_pageviews[
        website_pageviews['pageview_url'] == '/thank-you-for-your-order'
    ]['website_session_id'].unique()

    session_times['is_converted'] = session_times['website_session_id'].isin(thank_you_sessions)

    # Plot the funnel: distribution of session durations (boxplot)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=session_times, x='is_converted', y='session_duration_min', palette='coolwarm', ax=ax)
    plt.xlabel('Converted')
    plt.ylabel('Session Duration (min)')
    plt.xticks([0, 1], ['No', 'Yes'])
    st.pyplot(fig)




def render_session_funnel(session_times, website_pageviews):
    # Identify converted sessions
    thank_you_sessions = website_pageviews[website_pageviews['pageview_url'] == '/thank-you-for-your-order']['website_session_id'].unique()
    session_times['is_converted'] = session_times['website_session_id'].isin(thank_you_sessions)

    # Plot session duration vs conversion
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=session_times, x='is_converted', y='session_duration_min', palette='coolwarm', ax=ax)
    plt.xlabel('Converted')
    plt.ylabel('Session Duration (min)')
    plt.title("Session Duration vs Conversion")
    plt.xticks([0, 1], ['No', 'Yes'])
    st.pyplot(fig)

    # Summary table
    st.subheader("Session Duration Stats")
    summary = session_times.groupby('is_converted')['session_duration_min'].describe().reset_index()
    summary['is_converted'] = summary['is_converted'].map({True: 'Converted', False: 'Not Converted'})
    st.dataframe(summary)


def render_conversion_funnel_steps(website_pageviews):
    funnel_steps = [
        '/home',
        '/category',
        '/product',
        '/cart',
        '/checkout',
        '/thank-you-for-your-order'
    ]

    funnel_data = []
    for i, step in enumerate(funnel_steps):
        current_sessions = website_pageviews[website_pageviews['pageview_url'] == step]['website_session_id'].nunique()
        if i == 0:
            conversion_rate = 100.0
        else:
            previous_sessions = website_pageviews[website_pageviews['pageview_url'] == funnel_steps[i - 1]]['website_session_id'].nunique()
            conversion_rate = round((current_sessions / previous_sessions) * 100, 2) if previous_sessions else 0.0

        funnel_data.append({
            'Step': step,
            'Sessions': current_sessions,
            'Conversion from Previous (%)': conversion_rate
        })

    funnel_df = pd.DataFrame(funnel_data)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=funnel_df, x='Step', y='Sessions', palette='Blues_d', ax=ax)
    plt.title("Website Conversion Funnel")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # Table
    st.subheader("Funnel Details")
    st.dataframe(funnel_df)











