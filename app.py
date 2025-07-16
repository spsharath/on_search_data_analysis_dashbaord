import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------
# 🔐 Supabase Config
# ---------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ACTIVE_STATUSES = ["mapped", "on_ondc", "temp_closed"]

# ---------------------------------------------
# Upload & Load Data
# ---------------------------------------------
def load_data(file):
    df = pd.read_excel(file)

    df['Created At'] = pd.to_datetime(df['Created At'], format="%d %b %Y %I:%M:%S %p", errors='coerce')
    df['Provider ID'] = df['Outlet Name'].str.extract(r"(\d{4,5}:\d{4,5})")[0]
    df['Provider ID'] = df['Provider ID'].astype(str).str.strip()
    df['Outlet Name Clean'] = df['Outlet Name'].str.replace(r"^\d{4,5}:\d{4,5}\s*", "", regex=True)
    df['Date'] = df['Created At'].dt.date

    upload_to_supabase(df)
    return df

# ---------------------------------------------
# Upload to Supabase with Deduplication & 7-day Cleanup
# ---------------------------------------------
def upload_to_supabase(df):
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
    supabase.table("search_data").delete().lt("created_at", seven_days_ago).execute()

    insert_df = df[[
        'Last Search Event', 'Outlet Name', 'Created At', 'City Code', 'Buyer App',
        'Account Name', 'Message', 'Status', 'Provider ID', 'Outlet Name Clean', 'Date'
    ]].dropna(subset=['Created At'])

    insert_df.columns = [
        'last_search_event', 'outlet_name', 'created_at', 'city_code', 'buyer_app',
        'account_name', 'message', 'status', 'provider_id', 'outlet_name_clean', 'date'
    ]

    insert_df = insert_df.drop_duplicates(subset=['last_search_event', 'outlet_name'], keep='last')
    insert_df['created_at'] = pd.to_datetime(insert_df['created_at'])
    insert_df['created_at'] = insert_df['created_at'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    insert_df['date'] = insert_df['date'].astype(str)
    insert_df = insert_df.replace([np.inf, -np.inf], np.nan)
    insert_df = insert_df.where(pd.notnull(insert_df), None)
    records = insert_df.to_dict(orient='records')

    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table("search_data")\
            .upsert(batch, on_conflict='last_search_event,outlet_name')\
            .execute()

# ---------------------------------------------
# Upload Store Master File to Supabase
# ---------------------------------------------
def upload_store_master(file):
    df = pd.read_excel(file)
    df.columns = [col.strip() for col in df.columns]

    if not {'Name', 'Status', 'Provider ID'}.issubset(set(df.columns)):
        st.error("Store master file missing one of the required columns: 'Name', 'Status', 'Provider ID'")
        st.stop()

    df = df[['Name', 'Status', 'Provider ID']].dropna(subset=['Provider ID'])
    df['Provider ID'] = df['Provider ID'].astype(str).str.strip()
    df['uploaded_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    df['store_id'] = df['Provider ID']

    supabase.table("store_master").delete().neq("provider_id", "").execute()

    insert_df = df.replace([np.inf, -np.inf], np.nan)
    insert_df = insert_df.where(pd.notnull(insert_df), None)
    insert_df = insert_df.rename(columns={'Provider ID': 'provider_id', 'Name': 'name', 'Status': 'status'})
    records = insert_df.to_dict(orient='records')
    for i in range(0, len(records), 1000):
        supabase.table("store_master").insert(records[i:i+1000]).execute()

    return insert_df['uploaded_at'].iloc[0]

# ---------------------------------------------
# Load Store Master Data from Supabase
# ---------------------------------------------
@st.cache_data
def load_store_master():
    response = supabase.table("store_master").select("*").execute()
    data = response.data
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.columns = [col.lower() for col in df.columns]
    if 'uploaded_at' in df.columns:
        df['uploaded_at'] = pd.to_datetime(df['uploaded_at'], errors='coerce')
    df['provider_id'] = df['provider_id'].astype(str).str.strip()
    return df

# ---------------------------------------------
# Load Search Data from Supabase (Pagination)
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

    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    # Create a Date column for filtering & analysis
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    df.rename(columns={
        'last_search_event': 'Last Search Event',
        'outlet_name': 'Outlet Name',
        'created_at': 'Created At',
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
# UI Setup
# ---------------------------------------------
st.set_page_config(page_title="Search Data Analysis Dashboard", layout="wide")
st.title("Search Data Analysis Dashboard")

source_option = st.sidebar.radio("📦 Data Source", ["Upload Excel", "Load from Supabase"])

if source_option == "Upload Excel":
    uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])
    if uploaded_file is None:
        st.warning("Please upload an Excel file.")
        st.stop()
    df = load_data(uploaded_file)
else:
    df = load_from_supabase()

# Ensure 'Date' column exists
if 'Date' not in df.columns:
    if 'Created At' in df.columns:
        df['Date'] = pd.to_datetime(df['Created At'], errors='coerce').dt.date
    else:
        st.error("The data does not contain a 'Date' or 'Created At' column. Please check your input file or database.")
        st.stop()

# -- IMPORTANT: Define df_latest here --
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

# Update data type selection radio and logic for Tab 1

data_type = st.sidebar.radio(
    "Select data type for dashboard views:",
    ["Store Unique Data", "Unique Data Based on Event ID", "Raw Data"],
    index=0
)

if data_type == "Store Unique Data":
    # Unique by (Date, Buyer App, Provider ID)
    if not all(col in df.columns for col in ["Date", "Buyer App", "Provider ID"]):
        st.error("Missing required columns for Store Unique Data view.")
        st.stop()
    df_to_use = df.drop_duplicates(subset=["Date", "Buyer App", "Provider ID"], keep="last")
elif data_type == "Unique Data Based on Event ID":
    # Unique by (Last Search Event, Outlet Name)
    df_to_use = df.sort_values('Created At').drop_duplicates(
        subset=['Last Search Event', 'Outlet Name'], keep='last'
    )
else:
    df_to_use = df

# ---------------------------------------------
# Helper: Prepare Comparison Data
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
# Dashboard Tabs
# ---------------------------------------------

buyer_apps_all = sorted(df['Buyer App'].dropna().unique())
city_codes_all = sorted(df['City Code'].dropna().unique())

tab1, tab2, tab3 = st.tabs([
    "Buyer App Comparison", "All Days Summary",
    "Store Coverage"
])

with tab1:
    st.subheader(f"Buyer App Counts: {yesterday_str} vs {today_str}")
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

    st.subheader("📥 Missing Stores Report (Store Master vs Store Data)")

    # Load store master data (from DB)
    store_master_df = load_store_master()
    active_statuses = ["mapped", "on_ondc", "temp_closed"]
    active_stores = store_master_df[store_master_df['status'].str.lower().isin(active_statuses)].copy()
    active_stores['provider_id'] = active_stores['provider_id'].astype(str).str.strip()
    active_stores = active_stores[active_stores['provider_id'].notna() & (active_stores['provider_id'] != 'nan')]
    active_pids = set(active_stores['provider_id'])

    # Prepare filters
    available_dates = sorted(df['Date'].dropna().unique())
    available_buyer_apps = sorted(df['Buyer App'].dropna().astype(str).str.strip().unique())

    col1, col2 = st.columns([1, 1])
    with col1:
        select_all_dates = st.button("Select All Dates", key="msr_select_all_dates")
        clear_all_dates = st.button("Clear All Dates", key="msr_clear_all_dates")
        if 'msr_selected_dates' not in st.session_state:
            st.session_state.msr_selected_dates = available_dates.copy()
        if select_all_dates:
            st.session_state.msr_selected_dates = available_dates.copy()
        if clear_all_dates:
            st.session_state.msr_selected_dates = []
        selected_dates = st.multiselect(
            "Choose Dates:",
            available_dates,
            default=st.session_state.msr_selected_dates,
            key="msr_dates_multiselect"
        )
        st.session_state.msr_selected_dates = selected_dates
    with col2:
        select_all_buyers = st.button("Select All Buyer Apps", key="msr_select_all_buyers")
        clear_all_buyers = st.button("Clear All Buyer Apps", key="msr_clear_all_buyers")
        if 'msr_selected_buyers' not in st.session_state:
            st.session_state.msr_selected_buyers = available_buyer_apps.copy()
        if select_all_buyers:
            st.session_state.msr_selected_buyers = available_buyer_apps.copy()
        if clear_all_buyers:
            st.session_state.msr_selected_buyers = []
        selected_buyer_apps = st.multiselect(
            "Choose Buyer Apps:",
            available_buyer_apps,
            default=st.session_state.msr_selected_buyers,
            key="msr_buyers_multiselect"
        )
        st.session_state.msr_selected_buyers = selected_buyer_apps

    # Prepare missing stores data
    missing_counts = []
    missing_rows = []
    for date in selected_dates:
        for buyer_app in selected_buyer_apps:
            # If Provider ID column is missing in search data, treat all as missing
            if 'Provider ID' not in df.columns:
                sent_pids = set()
            else:
                filtered = df[
                    (df['Date'] == date) &
                    (df['Buyer App'].astype(str).str.strip() == str(buyer_app).strip())
                ].copy()
                filtered['Provider ID'] = filtered['Provider ID'].astype(str).str.strip()
                filtered = filtered[filtered['Provider ID'].notna() & (filtered['Provider ID'] != 'nan')]
                sent_pids = set(filtered['Provider ID'])
            missing_pids = active_pids - sent_pids
            missing_count = len(missing_pids)
            missing_counts.append({
                'Buyer App': buyer_app,
                'Date': date,
                'Missing Count': missing_count
            })
            for pid in missing_pids:
                store_row = active_stores[active_stores['provider_id'] == pid].iloc[0]
                missing_rows.append({
                    'Date': date,
                    'Buyer App': buyer_app,
                    'Provider ID': pid,
                    'Store Name': store_row['name'],
                    'Status': store_row['status']
                })

    # Bar graph (descending order by missing count)
    if missing_counts:
        missing_counts_df = pd.DataFrame(missing_counts)
        # Group by Buyer App (sum across selected dates)
        bar_df = missing_counts_df.groupby('Buyer App')['Missing Count'].sum().reset_index()
        bar_df = bar_df.sort_values('Missing Count', ascending=False)
        fig = px.bar(bar_df, x='Buyer App', y='Missing Count', title="Missing Active Stores per Buyer App (sum across selected dates)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing stores for the selected filters.")

    # Table of missing stores
    if missing_rows:
        missing_stores_df = pd.DataFrame(missing_rows)
        st.dataframe(missing_stores_df)
        st.download_button(
            "📥 Download Missing Stores Report",
            data=missing_stores_df.to_csv(index=False),
            file_name="missing_stores_report.csv",
            mime='text/csv'
        )
    else:
        st.info("No missing stores to report for the selected filters.")


with tab2:
    st.subheader("All Days Summary with Filters")

    with st.expander("Filter Options", expanded=True):
        col1, col2 = st.columns([1, 1])

        if 'selected_buyers' not in st.session_state:
            st.session_state.selected_buyers = buyer_apps_all.copy()
        if 'selected_cities' not in st.session_state:
            st.session_state.selected_cities = city_codes_all.copy()

        with col1:
            st.markdown("**Buyer Apps**")
            select_all_buyers = st.button("Select All Buyers")
            clear_all_buyers = st.button("Clear All Buyers")
            if select_all_buyers:
                st.session_state.selected_buyers = buyer_apps_all.copy()
            if clear_all_buyers:
                st.session_state.selected_buyers = []
                valid_selected_buyers = [b for b in st.session_state.selected_buyers if b in buyer_apps_all]
            selected_buyer_apps = st.multiselect(
                "Choose Buyer Apps:",
                buyer_apps_all,
                default=st.session_state.selected_buyers,
                key="buyer_multiselect"
            )
            st.session_state.selected_buyers = selected_buyer_apps

        with col2:
            st.markdown("**City Codes**")
            select_all_cities = st.button("Select All Cities")
            clear_all_cities = st.button("Clear All Cities")
            if select_all_cities:
                st.session_state.selected_cities = city_codes_all.copy()
            if clear_all_cities:
                st.session_state.selected_cities = []
                valid_selected_cities = [c for c in st.session_state.selected_cities if c in city_codes_all]
            selected_city_codes = st.multiselect(
                "Choose City Codes:",
                city_codes_all,
                default=st.session_state.selected_cities,
                key="city_multiselect"
            )
            st.session_state.selected_cities = selected_city_codes

    filtered_df = df[
        (df['Buyer App'].isin(selected_buyer_apps)) &
        (df['City Code'].isin(selected_city_codes))
    ]

    st.write(f"Showing {len(filtered_df):,} records after filtering")

    summary = filtered_df.groupby(['Date', 'Buyer App', 'City Code']).size().reset_index(name='Count')
    pivot = summary.pivot_table(index='Date', columns=['Buyer App', 'City Code'], values='Count', fill_value=0)
    st.dataframe(summary)

    total_counts = summary.groupby('Date')['Count'].sum().reset_index()
    fig = px.bar(total_counts, x='Date', y='Count', title="Total Records per Date (Filtered)")
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.subheader("Store Coverage: Active Stores NOT Sent to Buyer Apps")

    store_file = st.file_uploader("\U0001F4E4 Upload Store Master File", type=["xlsx"], key="store_master_upload")
    if store_file:
        uploaded_at = upload_store_master(store_file)
        st.success(f"✅ Store master uploaded at {uploaded_at}")
        store_df = load_store_master()
    else:
        store_df = load_store_master()
        if store_df.empty:
            st.warning("⚠️ No store master uploaded yet.")
            st.stop()
        else:
            uploaded_at = store_df['uploaded_at'].max()
            st.info(f"Last uploaded: {uploaded_at.strftime('%d-%B %Y %I:%M %p')}")

    if not store_df.empty:
        active_stores = store_df[store_df['status'].str.lower().isin(ACTIVE_STATUSES)]
        active_pids = set(active_stores['provider_id'].astype(str).str.strip())

        available_dates_tab3 = sorted(df_latest['Date'].dropna().unique())
        selected_date_tab3 = st.selectbox("\U0001F4C5 Select Date for Coverage Analysis:", available_dates_tab3, key="tab3_date")

        df_selected_date = df_latest[df_latest['Date'] == selected_date_tab3]

        buyer_apps_from_data = set(df_selected_date['Buyer App'].dropna().astype(str).str.strip().unique())
        buyer_apps_from_master = set(df_latest['Buyer App'].dropna().astype(str).str.strip().unique())
        buyer_apps_all = sorted(buyer_apps_from_data.union(buyer_apps_from_master))
        coverage_data = []

        for buyer_app in buyer_apps_all:
            buyer_app_str = str(buyer_app).strip()
            buyer_df = df_selected_date[df_selected_date['Buyer App'].astype(str).str.strip() == buyer_app_str]
            if not buyer_df.empty:
                sent_pids_buyer = set(buyer_df['Provider ID'].dropna().astype(str).str.strip())
            else:
                sent_pids_buyer = set()
            not_sent_count = len(active_pids - sent_pids_buyer)
            coverage_data.append({
                "Buyer App": buyer_app_str,
                "Active but not Sent": not_sent_count
            })

        coverage_df = pd.DataFrame(coverage_data)
        coverage_df = coverage_df.sort_values(by='Active but not Sent', ascending=False)

        if coverage_df['Active but not Sent'].sum() > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=coverage_df['Buyer App'],
                y=coverage_df['Active but not Sent'],
                text=coverage_df['Active but not Sent'],
                textposition='auto',
                marker_color='indianred'
            ))
            fig.update_layout(
                title=f"Active Stores NOT Sent to Buyer Apps (on {selected_date_tab3})",
                xaxis_title="Buyer App",
                yaxis_title="Count of Active Stores NOT Sent",
                template='plotly_white',
                yaxis=dict(tick0=0, dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ All active stores have been sent to all buyer apps for the selected date.")

        st.subheader("Not Sent Stores Table (Filtered)")
        selected_buyer_tab3 = st.selectbox("Select Buyer App:", buyer_apps_all, key="tab3_buyer")
        selected_buyer_tab3_str = str(selected_buyer_tab3).strip()
        filtered_df = df_selected_date[
            df_selected_date['Buyer App'].astype(str).str.strip() == selected_buyer_tab3_str
        ]
        sent_pids_filtered = set(filtered_df['Provider ID'].dropna().astype(str).str.strip())
        sent_pids_filtered = set(pid for pid in sent_pids_filtered if pid and pid != 'nan')
        missing_pids = active_pids - sent_pids_filtered
        not_sent_df = active_stores[active_stores['provider_id'].astype(str).str.strip().isin(missing_pids)]

        st.warning(f"⚠️ {len(not_sent_df)} active stores not sent to ONDC for {selected_buyer_tab3_str} on {selected_date_tab3}")
        st.dataframe(not_sent_df[['name', 'provider_id', 'status']])
        st.download_button(
            "\U0001F4C5 Download Missing Active Stores",
            data=not_sent_df.to_csv(index=False),
            file_name=f"active_not_sent_to_ondc_{selected_buyer_tab3_str}_{selected_date_tab3}.csv",
            mime="text/csv"
        )

