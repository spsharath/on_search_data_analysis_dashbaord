import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import numpy as np

# ---------------------------------------------
# 🔐 Supabase Config (set via secrets.toml)
# ---------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------
# 🔁 Load Data from Excel Upload
# ---------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    # Parse Created At with fallback coercion
    df['Created At'] = pd.to_datetime(
        df['Created At'], format="%d %b %Y %I:%M:%S %p", errors='coerce'
    )

    # Extract Provider ID and Outlet Name Clean from Outlet Name string
    df[['Provider ID', 'Outlet Name Clean']] = df['Outlet Name'].str.extract(r"(\d{4,5}:\d{4,5})\s*(.*)")

    # Extract Date from Created At
    df['Date'] = df['Created At'].dt.date

    # Upload to Supabase
    upload_to_supabase(df)

    return df

# ---------------------------------------------
# 💾 Upload Data to Supabase (Insert Only)
# ---------------------------------------------
def upload_to_supabase(df):
    insert_df = df[[
        'Last Search Event', 'Outlet Name', 'Created At', 'City Code', 'Buyer App',
        'Account Name', 'Message', 'Status', 'Provider ID', 'Outlet Name Clean', 'Date'
    ]].dropna(subset=['Created At'])

    # Rename columns to match DB schema
    insert_df.columns = [
        'last_search_event', 'outlet_name', 'created_at', 'city_code', 'buyer_app',
        'account_name', 'message', 'status', 'provider_id', 'outlet_name_clean', 'date'
    ]

    # Format datetime columns as ISO strings
    insert_df['created_at'] = insert_df['created_at'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    insert_df['date'] = insert_df['date'].astype(str)

    # Replace NaN and infinite with None for JSON compatibility
    insert_df = insert_df.replace([np.inf, -np.inf], np.nan)
    insert_df = insert_df.where(pd.notnull(insert_df), None)

    records = insert_df.to_dict(orient='records')

    # Insert in batches of 1000
    for i in range(0, len(records), 1000):
        batch = records[i:i+1000]
        supabase.table("search_data").insert(batch).execute()

# ---------------------------------------------
# 🔁 Load Data from Supabase with Pagination
# ---------------------------------------------
@st.cache_data
def load_from_supabase():
    all_rows = []
    batch_size = 1000
    offset = 0

    while True:
        response = supabase.table("search_data")\
            .select("*")\
            .range(offset, offset + batch_size - 1)\
            .execute()

        batch = response.data
        if not batch:
            break

        all_rows.extend(batch)
        offset += batch_size

    df = pd.DataFrame(all_rows)

    # Parse dates safely
    df['Created At'] = pd.to_datetime(df['created_at'], format='ISO8601', errors='coerce')
    df['Date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    # Rename columns to user-friendly names
    df.rename(columns={
        'last_search_event': 'Last Search Event',
        'outlet_name': 'Outlet Name',
        'city_code': 'City Code',
        'buyer_app': 'Buyer App',
        'account_name': 'Account Name',
        'message': 'Message',
        'status': 'Status',
        'provider_id': 'Provider ID',
        'outlet_name_clean': 'Outlet Name Clean'
    }, inplace=True)

    st.success(f"✅ Loaded {len(df):,} records from Supabase")

    return df

# ---------------------------------------------
# 🧭 UI Setup
# ---------------------------------------------
st.set_page_config(page_title="Search Data Analysis Dashboard", layout="wide")
st.title("🔍 Search Data Analysis Dashboard")

source_option = st.sidebar.radio("📦 Data Source", ["Upload Excel", "Load from Supabase"])

if source_option == "Upload Excel":
    uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])
    if uploaded_file is None:
        st.warning("Please upload an Excel file.")
        st.stop()
    df = load_data(uploaded_file)
else:
    df = load_from_supabase()

# ---------------------------------------------
# 🧼 Data Preparation
# ---------------------------------------------
df_latest = df.sort_values('Created At').drop_duplicates(
    subset=['Last Search Event', 'Outlet Name'], keep='last'
)

latest_dates = sorted(df['Date'].dropna().unique())[-2:]
if len(latest_dates) < 2:
    st.error("Not enough data for two days.")
    st.stop()

yesterday, today = latest_dates
yesterday_str = pd.to_datetime(yesterday).strftime("%d-%B")
today_str = pd.to_datetime(today).strftime("%d-%B")

data_type = st.sidebar.radio(
    "Select data type for dashboard views:",
    ["Unique Data", "Raw Data"],
    index=0
)

df_to_use = df_latest if data_type == "Unique Data" else df

# ---------------------------------------------
# 📊 Helper: Prepare Comparison Data
# ---------------------------------------------
def prepare_comparison(df, group_by):
    df_yesterday = df[df['Date'] == yesterday]
    df_today = df[df['Date'] == today]
    all_values = pd.concat([df_yesterday[group_by], df_today[group_by]]).drop_duplicates()
    counts_yesterday = df_yesterday.groupby(group_by).size().reindex(all_values).reset_index()
    counts_today = df_today.groupby(group_by).size().reindex(all_values).reset_index()
    counts_yesterday.columns = [group_by, 'Count']
    counts_today.columns = [group_by, 'Count']
    counts_yesterday['Date'] = yesterday_str
    counts_today['Date'] = today_str
    counts = pd.concat([counts_yesterday, counts_today])
    counts[group_by] = counts[group_by].fillna('Unknown')
    counts['Count'] = counts['Count'].fillna(0).astype(int)
    return counts

# ---------------------------------------------
# 📊 Dashboard Tabs
# ---------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Buyer App Comparison", "City Code Comparison",
    "prod.nirmitbap.ondc.org Std:01 Graph", "Inspect & Download Data"
])

with tab1:
    st.subheader(f"📊 Buyer App Counts: {yesterday_str} vs {today_str}")
    st.markdown(f"**Data Mode:** {data_type}")
    buyer_counts = prepare_comparison(df_to_use, 'Buyer App')
    fig1 = px.bar(buyer_counts, x='Buyer App', y='Count', color='Date', barmode='group')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📥 Missing Stores Report")
    buyer_apps = sorted(df_to_use['Buyer App'].dropna().unique())
    selected_buyer = st.selectbox("Select Buyer App:", buyer_apps)
    df_y = df_to_use[(df_to_use['Date'] == yesterday) & (df_to_use['Buyer App'] == selected_buyer)]
    df_t = df_to_use[(df_to_use['Date'] == today) & (df_to_use['Buyer App'] == selected_buyer)]
    df_y = df_y[['Outlet Name', 'Provider ID', 'City Code']].dropna()
    df_t = df_t[['Outlet Name', 'Provider ID', 'City Code']].dropna()
    missing = set(df_y.apply(tuple, axis=1)) - set(df_t.apply(tuple, axis=1))
    if not missing:
        st.success("✅ No missing stores. All records from yesterday are present today.")
    else:
        missing_df = pd.DataFrame(list(missing), columns=["Outlet Name", "Provider ID", "City Code"])
        st.warning(f"⚠️ {len(missing_df)} missing stores")
        st.dataframe(missing_df)
        st.download_button(
            "📤 Download Missing Report",
            data=missing_df.to_csv(index=False),
            file_name=f"{selected_buyer}_missing_{yesterday_str}_not_in_{today_str}.csv",
            mime='text/csv')

with tab2:
    st.subheader(f"📊 City Code Counts: {yesterday_str} vs {today_str}")
    st.markdown(f"**Data Mode:** {data_type}")
    city_counts = prepare_comparison(df_to_use, 'City Code')
    fig2 = px.bar(city_counts, x='City Code', y='Count', color='Date', barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("📊 prod.nirmitbap.ondc.org — With & Without std:01")
    df_buyer = df_to_use[df_to_use['Buyer App'] == "prod.nirmitbap.ondc.org"]

    def count_std_filter(day, std01=True):
        df_day = df_buyer[df_buyer['Date'] == day]
        if std01:
            df_day = df_day[df_day['City Code'].str.startswith("std:01")]
        else:
            df_day = df_day[~df_day['City Code'].str.startswith("std:01")]
        df_day = df_day.sort_values('Created At').drop_duplicates(['Outlet Name', 'City Code'])
        return len(df_day)

    summary = []
    for d in [yesterday, today]:
        lbl = pd.to_datetime(d).strftime("%d-%B")
        summary.append({"Date": lbl, "Type": "With std:01", "Count": count_std_filter(d, True)})
        summary.append({"Date": lbl, "Type": "Without std:01", "Count": count_std_filter(d, False)})

    fig3 = px.bar(pd.DataFrame(summary), x='Type', y='Count', color='Date', barmode='group')
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("📋 Inspect & Download Data")
    st.markdown(f"**Data Mode:** {data_type}")
    col1, col2 = st.columns(2)

    with col1:
        sel_buyer = st.selectbox("Select Buyer App:", df_to_use['Buyer App'].dropna().unique())
        df_b = df_to_use[df_to_use['Buyer App'] == sel_buyer]
        st.dataframe(df_b)
        st.download_button("📥 Download Buyer App Data", data=df_b.to_csv(index=False),
                           file_name=f"{sel_buyer}_{data_type.replace(' ', '_')}.csv", mime='text/csv')

    with col2:
        sel_city = st.selectbox("Select City Code:", df_to_use['City Code'].dropna().unique())
        df_c = df_to_use[df_to_use['City Code'] == sel_city]
        st.dataframe(df_c)
        st.download_button("📥 Download City Code Data", data=df_c.to_csv(index=False),
                           file_name=f"{sel_city}_{data_type.replace(' ', '_')}.csv", mime='text/csv')
