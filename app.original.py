import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Search Data Analysis Dashboard", layout="wide")

st.title("🔍 Search Data Analysis Dashboard")

# 📥 Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to continue.")
    st.stop()

# 📄 Read data
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    # fix Created At datetime
    df['Created At'] = pd.to_datetime(
        df['Created At'],
        format="%d %b %Y %I:%M:%S %p",
        errors='coerce'
    )
    # split Outlet Name into Provider ID & Name
    df[['Provider ID', 'Outlet Name Clean']] = df['Outlet Name'].str.extract(r"(\d{4,5}:\d{4,5})\s*(.*)")
    return df

df = load_data(uploaded_file)

st.subheader("📋 Raw Data (first 10 rows)")
st.dataframe(df.head(10))

# 📈 Deduplicate — latest per (Last Search Event + Outlet Name)
df_latest = df.sort_values('Created At').drop_duplicates(
    subset=['Last Search Event', 'Outlet Name'], keep='last'
)

st.subheader("✅ Deduplicated Data (latest per Last Search Event & Outlet)")
st.write(f"Total rows after deduplication: {len(df_latest)}")
st.dataframe(df_latest.head(10))

# 📊 Analysis: Counts by City Code
st.header("📊 Analysis Reports")

tab1, tab2, tab3 = st.tabs(["City Code", "Buyer App", "Account Name"])

with tab1:
    city_counts = df_latest['City Code'].value_counts().reset_index()
    city_counts.columns = ['City Code', 'Count']
    fig1 = px.bar(city_counts, x='City Code', y='Count', title="Counts by City Code")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    buyer_counts = df_latest['Buyer App'].value_counts().reset_index()
    buyer_counts.columns = ['Buyer App', 'Count']
    fig2 = px.bar(buyer_counts, x='Buyer App', y='Count', title="Counts by Buyer App")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    account_counts = df_latest['Account Name'].value_counts().reset_index()
    account_counts.columns = ['Account Name', 'Count']
    fig3 = px.bar(account_counts, x='Account Name', y='Count', title="Counts by Account Name")
    st.plotly_chart(fig3, use_container_width=True)

# 🎯 Special case: prod.nirmitbap.ondc.org + std:01*
st.header("🎯 Special Analysis: `prod.nirmitbap.ondc.org` with `std:01*` City Codes")

df_special = df[
    (df['Buyer App'] == "prod.nirmitbap.ondc.org") &
    (df['City Code'].str.startswith("std:01"))
].copy()

if df_special.empty:
    st.info("No data found for prod.nirmitbap.ondc.org with City Code starting with std:01.")
else:
    df_special = df_special.sort_values('Created At').drop_duplicates(
        subset=['Outlet Name'], keep='last'
    )

    st.write(f"Total unique outlets (latest per Outlet) for prod.nirmitbap.ondc.org in std:01*: {len(df_special)}")

    fig_special = px.bar(
        df_special['City Code'].value_counts().reset_index(),
        x='index', y='City Code',
        labels={'index': 'City Code', 'City Code': 'Count'},
        title="prod.nirmitbap.ondc.org — Latest Outlets per City (std:01*)"
    )
    st.plotly_chart(fig_special, use_container_width=True)

    st.dataframe(df_special.head(10))

