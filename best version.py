import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Search Data Analysis Dashboard", layout="wide")

st.title("🔍 Search Data Analysis Dashboard")

uploaded_file = st.file_uploader("📥 Upload Excel file", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file.")
    st.stop()


@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    df['Created At'] = pd.to_datetime(
        df['Created At'],
        format="%d %b %Y %I:%M:%S %p",
        errors='coerce'
    )

    df[['Provider ID', 'Outlet Name Clean']] = df['Outlet Name'].str.extract(r"(\d{4,5}:\d{4,5})\s*(.*)")

    df['Date'] = df['Created At'].dt.date
    return df


df = load_data(uploaded_file)

# Deduplicate on Last Search Event & Outlet Name (latest only)
df_latest = df.sort_values('Created At').drop_duplicates(
    subset=['Last Search Event', 'Outlet Name'], keep='last'
)

latest_dates = sorted(df['Date'].dropna().unique())[-2:]
if len(latest_dates) < 2:
    st.error("Not enough data for two days.")
    st.stop()

yesterday, today = latest_dates

def prepare_comparison(df, group_by):
    df_yesterday = df[df['Date'] == yesterday]
    df_today = df[df['Date'] == today]

    all_values = pd.concat([
        df_yesterday[group_by],
        df_today[group_by]
    ]).drop_duplicates()

    counts_yesterday = df_yesterday.groupby(group_by).size().reindex(all_values).reset_index()
    counts_yesterday.columns = [group_by, 'Count']
    counts_yesterday['Date'] = 'Yesterday'

    counts_today = df_today.groupby(group_by).size().reindex(all_values).reset_index()
    counts_today.columns = [group_by, 'Count']
    counts_today['Date'] = 'Today'

    counts = pd.concat([counts_yesterday, counts_today])

    counts[group_by] = counts[group_by].fillna('Unknown')
    counts['Count'] = counts['Count'].fillna(0).astype(int)

    return counts


tab1, tab2, tab3, tab4 = st.tabs([
    "Buyer App Comparison",
    "City Code Comparison",
    "prod.nirmitbap.ondc.org Std:01 Graph",
    "Inspect & Download Data"
])

with tab1:
    st.subheader(f"📊 Buyer App Counts: {yesterday} vs {today}")
    buyer_counts = prepare_comparison(df_latest, 'Buyer App')
    fig1 = px.bar(
        buyer_counts,
        x='Buyer App', y='Count', color='Date',
        barmode='group', title='Buyer App Comparison'
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader(f"📊 City Code Counts: {yesterday} vs {today}")
    city_counts = prepare_comparison(df_latest, 'City Code')
    fig2 = px.bar(
        city_counts,
        x='City Code', y='Count', color='Date',
        barmode='group', title='City Code Comparison'
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("📊 prod.nirmitbap.ondc.org — Yesterday vs Today — With & Without std:01")

    df_buyer = df_latest[df_latest['Buyer App'] == "prod.nirmitbap.ondc.org"]

    def count_with_without(day, std01=True):
        df_day = df_buyer[df_buyer['Date'] == day].copy()
        if std01:
            df_day = df_day[df_day['City Code'].str.startswith("std:01")]
        else:
            df_day = df_day[~df_day['City Code'].str.startswith("std:01")]
        df_day_latest = df_day.sort_values('Created At').drop_duplicates(
            subset=['Outlet Name', 'City Code'], keep='last'
        )
        return len(df_day_latest)

    summary_data = []

    for day, label in zip([yesterday, today], ['Yesterday', 'Today']):
        summary_data.append({
            'Date': label,
            'Type': 'With std:01',
            'Count': count_with_without(day, std01=True)
        })
        summary_data.append({
            'Date': label,
            'Type': 'Without std:01',
            'Count': count_with_without(day, std01=False)
        })

    summary_df = pd.DataFrame(summary_data)

    fig4 = px.bar(
        summary_df,
        x='Type', y='Count', color='Date',
        barmode='group',
        title='prod.nirmitbap.ondc.org — With vs Without std:01 — Yesterday vs Today'
    )

    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.subheader("📋 Inspect & Download Data")

    # Toggle for Raw / Deduplicated
    data_type = st.radio(
        "Select data type:",
        ["Deduplicated Data", "Raw Data"],
        index=0,
        horizontal=True
    )

    if data_type == "Deduplicated Data":
        df_to_use = df_latest
    else:
        df_to_use = df

    col1, col2 = st.columns(2)

    with col1:
        selected_buyer = st.selectbox(
            "Select Buyer App to view detailed data:",
            df_to_use['Buyer App'].unique()
        )

        df_selected_buyer = df_to_use[df_to_use['Buyer App'] == selected_buyer]
        st.dataframe(df_selected_buyer)

        st.download_button(
            label=f"📥 Download {data_type} — Buyer App Data (CSV)",
            data=df_selected_buyer.to_csv(index=False),
            file_name=f"{selected_buyer}_{data_type.replace(' ', '_')}.csv",
            mime='text/csv'
        )

    with col2:
        selected_city = st.selectbox(
            "Select City Code to view detailed data:",
            df_to_use['City Code'].unique()
        )

        df_selected_city = df_to_use[df_to_use['City Code'] == selected_city]
        st.dataframe(df_selected_city)

        st.download_button(
            label=f"📥 Download {data_type} — City Code Data (CSV)",
            data=df_selected_city.to_csv(index=False),
            file_name=f"{selected_city}_{data_type.replace(' ', '_')}.csv",
            mime='text/csv'
        )
