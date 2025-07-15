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
yesterday_str = pd.to_datetime(yesterday).strftime("%d-%B")
today_str = pd.to_datetime(today).strftime("%d-%B")

# Global data toggle in sidebar
st.sidebar.subheader("🗂️ Data Type Selection")
data_type = st.sidebar.radio(
    "Select data type for dashboard views:",
    ["Unique Data", "Raw Data"],
    index=0
)

# Select DataFrame based on toggle
df_to_use = df_latest if data_type == "Unique Data" else df


def prepare_comparison(df, group_by):
    df_yesterday = df[df['Date'] == yesterday]
    df_today = df[df['Date'] == today]

    all_values = pd.concat([
        df_yesterday[group_by],
        df_today[group_by]
    ]).drop_duplicates()

    counts_yesterday = df_yesterday.groupby(group_by).size().reindex(all_values).reset_index()
    counts_yesterday.columns = [group_by, 'Count']
    counts_yesterday['Date'] = yesterday_str

    counts_today = df_today.groupby(group_by).size().reindex(all_values).reset_index()
    counts_today.columns = [group_by, 'Count']
    counts_today['Date'] = today_str

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
    st.subheader(f"📊 Buyer App Counts: {yesterday_str} vs {today_str}")
    st.markdown(f"**Data Mode:** {data_type}")

    # Plot buyer comparison chart
    buyer_counts = prepare_comparison(df_to_use, 'Buyer App')
    fig1 = px.bar(
        buyer_counts,
        x='Buyer App', y='Count', color='Date',
        barmode='group', title='Buyer App Comparison'
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Missing Stores Report")

    buyer_apps = sorted(df_to_use['Buyer App'].dropna().unique())

    selected_buyer_for_missing = st.selectbox(
        "Select Buyer App to generate report (Sent Yesterday but NOT Today):",
        buyer_apps
    )

    # Filter yesterday and today data for selected buyer app
    df_yesterday = df_to_use[
        (df_to_use['Date'] == yesterday) & (df_to_use['Buyer App'] == selected_buyer_for_missing)
    ][['Outlet Name', 'Provider ID', 'City Code']].dropna()

    df_today = df_to_use[
        (df_to_use['Date'] == today) & (df_to_use['Buyer App'] == selected_buyer_for_missing)
    ][['Outlet Name', 'Provider ID', 'City Code']].dropna()

    # Create sets for comparison
    yesterday_set = set(df_yesterday.apply(tuple, axis=1))
    today_set = set(df_today.apply(tuple, axis=1))

    missing_set = yesterday_set - today_set

    if not missing_set:
        st.success("✅ No missing stores. All records from yesterday are present today.")
    else:
        missing_df = pd.DataFrame(list(missing_set), columns=["Outlet Name", "Provider ID", "City Code"])
        st.warning(f"⚠️ {len(missing_df)} stores found in yesterday's data but missing today.")
        st.dataframe(missing_df)

        st.download_button(
            label="📤 Download Missing Stores Report (CSV)",
            data=missing_df.to_csv(index=False),
            file_name=f"{selected_buyer_for_missing.replace('.', '_')}_missing_{yesterday_str}_not_in_{today_str}.csv",
            mime='text/csv'
        )

with tab2:
    st.subheader(f"📊 City Code Counts: {yesterday_str} vs {today_str}")
    st.markdown(f"**Data Mode:** {data_type}")
    city_counts = prepare_comparison(df_to_use, 'City Code')
    fig2 = px.bar(
        city_counts,
        x='City Code', y='Count', color='Date',
        barmode='group', title='City Code Comparison'
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader(f"📊 prod.nirmitbap.ondc.org — {yesterday_str} vs {today_str} — With & Without std:01")
    st.markdown(f"**Data Mode:** {data_type}")

    df_buyer = df_to_use[df_to_use['Buyer App'] == "prod.nirmitbap.ondc.org"]

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

    for day in [yesterday, today]:
        label = pd.to_datetime(day).strftime("%d-%B")
        summary_data.append({
            'Date': label,
            'Type': 'Without std:01',
            'Count': count_with_without(day, std01=True)
        })
        summary_data.append({
            'Date': label,
            'Type': 'With std:01',
            'Count': count_with_without(day, std01=False)
        })

    summary_df = pd.DataFrame(summary_data)

    fig4 = px.bar(
        summary_df,
        x='Type', y='Count', color='Date',
        barmode='group',
        title='prod.nirmitbap.ondc.org — With vs Without std:01'
    )

    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.subheader("📋 Inspect & Download Data")
    st.markdown(f"**Data Mode:** {data_type}")

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
