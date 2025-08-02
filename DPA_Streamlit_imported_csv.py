import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import pyodbc
import os
from hashlib import sha256  # 🔒 For secure login

# --------------------------------------------
# 📌 Define the custom_kpi() layout function
# --------------------------------------------
def custom_kpi(label, value, help_text=None, column=None):
    """
    Renders a KPI-style metric box with optional help tooltip and custom column layout.
    """
    target = column if column else st
    with target.container():
        if help_text:
            target.markdown(f"**{label}** ℹ️")
            target.caption(help_text)
        else:
            target.markdown(f"**{label}**")
        target.markdown(f"<h3 style='color:#4CAF50'>{value}</h3>", unsafe_allow_html=True)


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
if st.session_state.get("logged_in", False):
    st.sidebar.title("Navigation")

    # Show logged-in user
    if st.session_state.get("user"):
        st.sidebar.markdown(f"👤 Logged in as: **{st.session_state.user.title()}**")

    # Dashboard options
    dashboard_options = [
        "Executive KPI Dashboard",
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
    ]

    # Maintain selected section using session state
    if "selected_section" not in st.session_state:
        st.session_state.selected_section = "Executive KPI Dashboard"

    selected_section = st.sidebar.radio("Go to", dashboard_options, index=dashboard_options.index(st.session_state.selected_section))
    st.session_state.selected_section = selected_section

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.selected_section = "Executive KPI Dashboard"
        st.rerun()

else:
    st.sidebar.title("🔐 Please log in")


# --------------------------


# 📁 CSV FILE LOADING FROM RELATIVE 'data/' FOLDER
import os
import pandas as pd
import streamlit as st

# ✅ Always load from ./data — works both locally and on Render
def get_data_path():
    return os.path.join(os.getcwd(), "data")

def load_data():
    try:
        data_path = get_data_path()

        orders = pd.read_csv(os.path.join(data_path, "orders.csv"))
        order_items = pd.read_csv(os.path.join(data_path, "order_items.csv"))
        order_item_refunds = pd.read_csv(os.path.join(data_path, "order_item_refunds.csv"))
        products = pd.read_csv(os.path.join(data_path, "products.csv"))
        website_pageviews = pd.read_csv(os.path.join(data_path, "website_pageviews.csv"))
        website_sessions = pd.read_csv(os.path.join(data_path, "website_sessions.csv"))

        st.success(f"✅ Loaded files from: {data_path}")
        return [orders, order_items, order_item_refunds, products, website_pageviews, website_sessions]
    except Exception as e:
        st.error(f"❌ Failed to load files: {e}")
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

# Convert 'created_at' columns to datetime
orders['created_at'] = pd.to_datetime(orders['created_at'])
order_items['created_at'] = pd.to_datetime(order_items['created_at'])
order_item_refunds['created_at'] = pd.to_datetime(order_item_refunds['created_at'])
products['created_at'] = pd.to_datetime(products['created_at'])
website_sessions['created_at'] = pd.to_datetime(website_sessions['created_at'])
website_pageviews['created_at'] = pd.to_datetime(website_pageviews['created_at'])

# Date filter (if you're using date inputs)

if "start_date" not in st.session_state or "end_date" not in st.session_state:
    min_date = orders["created_at"].min()
    max_date = orders["created_at"].max()
    st.session_state.start_date = min_date
    st.session_state.end_date = max_date

# Optional date input for user selection
st.sidebar.markdown("## 📅 Date Range Filter")
st.session_state.start_date = st.sidebar.date_input("Start Date", st.session_state.start_date)
st.session_state.end_date = st.sidebar.date_input("End Date", st.session_state.end_date)

start_date = pd.to_datetime(st.session_state.start_date)
end_date = pd.to_datetime(st.session_state.end_date)

# Filtered DataFrames
df_order_filtered = orders[(orders['created_at'] >= start_date) & (orders['created_at'] <= end_date)]
df_items_filtered = order_items[(order_items['created_at'] >= start_date) & (order_items['created_at'] <= end_date)]
df_refund_filtered = order_item_refunds[(order_item_refunds['created_at'] >= start_date) & (order_item_refunds['created_at'] <= end_date)]
df_products_filtered = products.copy()
df_filtered = website_sessions[(website_sessions['created_at'] >= start_date) & (website_sessions['created_at'] <= end_date)]
df_website_pageviews = website_pageviews[(website_pageviews['created_at'] >= start_date) & (website_pageviews['created_at'] <= end_date)]

# --- KPI CALCULATIONS ---



# ✅ Base metrics (use unfiltered unless necessary)
total_orders = orders['order_id'].nunique()
total_sessions = website_sessions['website_session_id'].nunique()
Total_Revenue = orders['price_usd'].sum()
Total_buyers = orders['user_id'].nunique()

# ✅ Net Revenue = All revenue - refunds
Net_Revenue = order_items['price_usd'].sum() - order_item_refunds['refund_amount_usd'].sum()

# ✅ Total cost = all item COGS
total_cost = order_items['cogs_usd'].sum()

# ✅ Net cost = only COGS from non-refunded items
# Filtered non-refunded items
non_refunded_items = order_items[~order_items['order_item_id'].isin(order_item_refunds['order_item_id'])]

# Net Revenue (excluding refunded)
Net_Revenue = non_refunded_items['price_usd'].sum()

# Net Cost (excluding refunded)
Net_cost = non_refunded_items['cogs_usd'].sum()

# Profit
profit = Net_Revenue - total_cost






# ✅ Total Items Sold 
Total_items = order_items.shape[0]  # All items, regardless of refund


# ✅ Bounce Rate (use unfiltered sessions, assuming 'is_bounce' exists and is 0/1)
Bounce_rate = website_sessions['is_bounce'].mean() * 100 if 'is_bounce' in website_sessions else 0

# ✅ Total Products
Total_Products = products['product_id'].nunique()

# ✅ Conversion Rate
Conversion_Rate = (total_orders / total_sessions) * 100 if total_sessions else 0

# ✅ Bounce Rate Handling

pageviews_per_session = website_pageviews.groupby('website_session_id').size()
single_page_sessions = pageviews_per_session[pageviews_per_session == 1].count()
total_sessions = website_sessions['website_session_id'].nunique()
bounce_rate = (single_page_sessions / total_sessions) * 100 if total_sessions else 0


# ✅ Averages
Avg_revenue_per_order = Total_Revenue / total_orders if total_orders else 0
Avg_revenue_per_buyer = Total_Revenue / Total_buyers if Total_buyers else 0
avg_profit_per_buyer = profit / Total_buyers if Total_buyers else 0
avg_item_per_order = Total_items / total_orders if total_orders else 0

# ✅ Refund Metrics
Total_Refund = order_item_refunds["refund_amount_usd"].sum()
Item_refunded_count = order_item_refunds['order_item_refund_id'].count()

# Total items sold before filtering refunds
total_items_sold_all = order_items.shape[0]

# Correct % Returned Items
pct_returned_items = (Item_refunded_count / total_items_sold_all) * 100


# ✅ One-time & Returning Buyers
buyers_by_order_count = orders.groupby('user_id')['order_id'].nunique()
one_time_buyers = buyers_by_order_count[buyers_by_order_count == 1].count()
returning_buyers = buyers_by_order_count[buyers_by_order_count > 1].count()
pct_one_time_buyers = (one_time_buyers / Total_buyers) * 100
pct_returning_buyers = (returning_buyers / Total_buyers) * 100

# ✅ User Sessions & Behavior
user_session_counts = website_sessions.groupby('user_id')['website_session_id'].nunique()
one_time_users = user_session_counts[user_session_counts == 1].count()
returning_users = user_session_counts[user_session_counts > 1].count()
avg_sessions_per_user = user_session_counts.mean()
pct_one_time_users = (one_time_users / user_session_counts.count()) * 100
pct_returning_users = (returning_users / user_session_counts.count()) * 100

# ✅ Days Between 1st & 2nd Purchase
orders_sorted = orders.sort_values(['user_id', 'created_at'])
first_two_orders = orders_sorted.groupby('user_id').head(2).copy()
first_two_orders['order_rank'] = first_two_orders.groupby('user_id').cumcount() + 1
pivot_orders = first_two_orders.pivot(index='user_id', columns='order_rank', values='created_at').dropna()
pivot_orders.columns = ['first_order', 'second_order']
pivot_orders['days_between'] = (pivot_orders['second_order'] - pivot_orders['first_order']).dt.days
avg_gap = pivot_orders['days_between'].mean()

# --------------------------
# 📊 KPI DASHBOARD (OVERVIEW)
# --------------------------
section = st.session_state.get("selected_section", "Executive KPI Dashboard")

if section == "Executive KPI Dashboard":
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()

    # Section 1: Core Metrics
    st.markdown("### 🔢 Core Metrics")
    st.markdown("### 🔢 Core Metrics")
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        custom_kpi("🔢 Total Orders", f"{total_orders:,}")
    with row1_col2:
        custom_kpi("🌐 Total Sessions", f"{total_sessions/1_000:,.0f}K")
    with row1_col3:
        custom_kpi("📈 Total Profit", f"${profit / 1_000_000:,.2f}M")

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        custom_kpi("💰 Total Revenue", f"${Total_Revenue / 1_000_000:,.2f}M")
    with row2_col2:
        custom_kpi("💸 Net Revenue", f"${Net_Revenue / 1_000_000:,.2f}M")
    with row2_col3:
        custom_kpi("📉 Total Cost (COGS)", f"${total_cost / 1_000:,.2f}K")

    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        custom_kpi("💸 Net Cost", f"${Net_cost/1000:,.2f}K")
    with row3_col2:
        custom_kpi("⚡ Conversion Rate", f"{Conversion_Rate:.2f}%")
    with row3_col3:
        custom_kpi("🚪 Bounce Rate", f"{bounce_rate:.2f}%")

    row10_col1, row10_col2, row10_col3 = st.columns(3)
    with row10_col1:
        custom_kpi("📦 Total Products", f"{Total_Products}")
    with row10_col2:
        custom_kpi("🛒 Total Items", f"{Total_items:,}")
    with row10_col3:
        custom_kpi("👥 Total Buyers", f"{Total_buyers:,}")

    # Section 2: Buyer Insights
    st.divider()
    st.markdown("### 🧑‍💼 Buyer Behavior")

    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        custom_kpi("💳 Avg Revenue/Buyer", f"${Avg_revenue_per_buyer:,.2f}")
    with row4_col2:
        custom_kpi("💹 Avg Profit/Buyer", f"${avg_profit_per_buyer:,.2f}")
    with row4_col3:
        custom_kpi("🛍️ Avg Revenue/Order", f"${Avg_revenue_per_order:,.2f}")

    row5_col1, row5_col2, row5_col3 = st.columns(3)
    with row5_col1:
        custom_kpi("🧾 One-time Buyers", f"{one_time_buyers:,}")
    with row5_col2:
        custom_kpi("🔁 Returning Buyers", f"{returning_buyers:,}")
    with row5_col3:
        custom_kpi("📈 % Returning Buyers", f"{pct_returning_buyers:.2f}%")

    row6_col1, row6_col2, row6_col3 = st.columns(3)
    with row6_col1:
        custom_kpi("📉 % One-time Buyers", f"{pct_one_time_buyers:.2f}%")
    with row6_col2:
        custom_kpi("📊 Avg Items/Order", f"{avg_item_per_order:.2f}")
    with row6_col3:
        custom_kpi("⏱️ Avg Days 1st → 2nd buy", f"{avg_gap:.2f} days")

    # Section 3: User Behavior
    st.divider()
    st.markdown("### 🧠 User Behavior")

    row7_col1, row7_col2, row7_col3 = st.columns(3)
    with row7_col1:
        custom_kpi("🧍 One-time Users", f"{one_time_users/1_000:,.2f}k")
    with row7_col2:
        custom_kpi("👥 Returning Users", f"{returning_users/1_000:,.2f}k")
    with row7_col3:
        custom_kpi("📈 Avg Sessions/User", f"{avg_sessions_per_user:.2f}")

    row8_col2, row8_col3 = st.columns(2)
    with row8_col2:
        custom_kpi("🧠 % Returning Users", f"{pct_returning_users:.2f}%")
    with row8_col3:
        custom_kpi("📉 % One-time Users", f"{pct_one_time_users:.2f}%")

    # Section 4: Refunds
    st.divider()
    st.markdown("### ↩️ Refund Metrics")

    row9_col1, row9_col2, row9_col3 = st.columns(3)
    with row9_col1:
        custom_kpi("💸 Total Refund Amount", f"${Total_Refund / 1_000:,.2f}k")
    with row9_col2:
        custom_kpi("📦 Total Items Refunded", f"{Item_refunded_count:,}")
    with row9_col3:
        custom_kpi("📉 % Returned Items", f"{pct_returned_items:.2f}%")

elif section == "Website Analytics":
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()

    st.title("🌐 Website Analytics")
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()

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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
    st.title("\U0001F4F1 Marketing Analytics")
    campaign_perf = website_sessions.groupby(['utm_source', 'utm_campaign']).agg(
        sessions=('website_session_id', 'count')
    ).reset_index()
    thank_you_sessions = website_pageviews[website_pageviews['pageview_url'] == '/thank-you-for-your-order']['website_session_id'].nunique()
    campaign_perf['conversions'] = thank_you_sessions // campaign_perf.shape[0]
    campaign_perf['conversion_rate'] = campaign_perf['conversions'] / campaign_perf['sessions'] * 100
    st.dataframe(campaign_perf)

elif section == "Customer Insights":
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access this section.")
        st.stop()
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













