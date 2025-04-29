import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Streamlit Setup
st.set_page_config(page_title="HCAHPS Self-Service Explorer", layout="wide")
st.title("📋 HCAHPS Self-Service Insights Dashboard")

sns.set(style="whitegrid")
plt.close('all')

# Load and Merge Data
github_base = "https://raw.githubusercontent.com/baheldeepti/Maven-Healthcare-Challenge-HCAHPS-dataset/main/data/"
files = {
    "state_results": "state_results.csv",
    "national_results": "national_results.csv",
    "measures": "measures.csv",
    "questions": "questions.csv",
    "responses": "responses.csv",
    "reports": "reports.csv",
    "states": "states.csv"
}
datasets = {name: pd.read_csv(github_base + filename) for name, filename in files.items()}

state_results_df = datasets['state_results']
national_results_df = datasets['national_results']
measures_df = datasets['measures']
questions_df = datasets['questions']
responses_df = datasets['responses']
reports_df = datasets['reports']
states_df = datasets['states']

responses_df['Response Rate (%)'] = pd.to_numeric(responses_df['Response Rate (%)'], errors='coerce')
for df in [state_results_df, national_results_df, responses_df]:
    df['Year'] = df['Release Period'].str[-4:].astype(int)

merged_measures_questions = pd.merge(measures_df, questions_df, on='Measure ID', how='left')
state_results_df = pd.merge(state_results_df, merged_measures_questions[['Measure ID', 'Measure', 'Question']], on='Measure ID', how='left')
national_results_df = pd.merge(national_results_df, merged_measures_questions[['Measure ID', 'Measure', 'Question']], on='Measure ID', how='left')
state_results_df = pd.merge(state_results_df, states_df[['State', 'State Name', 'Region']], on='State', how='left')
responses_df = pd.merge(responses_df, states_df[['State', 'State Name', 'Region']], on='State', how='left')

state_results_df.dropna(subset=['Question', 'Top-box Percentage', 'Measure'], inplace=True)
national_results_df.dropna(subset=['Question', 'Top-box Percentage', 'Measure'], inplace=True)

# Setup Tabs
tabs = st.tabs([
    "National Trends",
    "Most Improved Areas",
    "Score Disparities",
    "Regional Differences",
    "Response Rate Insights",
    "Opportunity Matrix",
    "AI Recommendations",
    "State-Level Comparison",
    "Patient Experience Heatmap",
    "Benchmarking Dashboard",
    "Anomaly Alerts"
])

# Tab 0: National Trends
with tabs[0]:
    st.subheader("📈 National Top-box % by Year and Measure")
    measure_year_trend = national_results_df.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=measure_year_trend, x='Year', y='Top-box Percentage', hue='Measure', ax=ax)
    ax.set_title("National Trends by Measure")
    ax.legend(title='Measure', bbox_to_anchor=(1.05, 1))
    ax.grid(True)
    st.pyplot(fig)

# Tab 1: Most Improved Areas
with tabs[1]:
    st.subheader("📊 Most Improved Questions")
    q_year = state_results_df.groupby(['Measure', 'Question', 'Year'])['Top-box Percentage'].mean().reset_index()
    pivot = q_year.pivot(index=['Measure', 'Question'], columns='Year', values='Top-box Percentage')

    if pivot.shape[1] >= 2:
        pivot['Improvement'] = pivot[pivot.columns[-1]] - pivot[pivot.columns[0]]
        improved = pivot.sort_values('Improvement', ascending=False).reset_index()
        top_improved = improved[['Measure', 'Question', 'Improvement']].head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_improved, y='Question', x='Improvement', hue='Measure', palette='viridis', ax=ax, dodge=False)
        st.pyplot(fig)
        st.dataframe(top_improved)
    else:
        st.info("Insufficient data to calculate improvement.")

# Tab 2: Score Disparities
with tabs[2]:
    st.subheader("📉 Most Declined Questions")
    if 'Improvement' in pivot.columns:
        declined = pivot[pivot['Improvement'] < 0].sort_values('Improvement').reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=declined.head(10), y='Question', x='Improvement', palette='flare', ax=ax, dodge=False)
        ax.set_title("Top 10 Declined Questions")
        st.pyplot(fig)

        st.markdown("### 📋 Details of Most Declined Questions")
        st.dataframe(declined[['Question', 'Improvement']].head(10))
    else:
        st.info("Insufficient year data to compute declines.")

# Tab 3: Regional Differences
with tabs[3]:
    st.subheader("🗺️ Regional Average Scores")
    reg_avg = state_results_df.groupby(['Region', 'Year'])['Top-box Percentage'].mean().reset_index()
    selected_year = st.slider("Select Year", int(reg_avg['Year'].min()), int(reg_avg['Year'].max()), int(reg_avg['Year'].max()), key='regional_year')
    chart_df = reg_avg[reg_avg['Year'] == selected_year]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=chart_df, x='Top-box Percentage', y='Region', palette='crest', ax=ax)
    ax.set_title(f"Regional Scores for {selected_year}")
    st.pyplot(fig)

# Tab 4: Response Rate Insights
with tabs[4]:
    st.subheader("📬 National & State Response Rate Trends")
    national_trend = responses_df.groupby('Year')["Response Rate (%)"].mean().reset_index()
    national_trend['Region/State'] = 'National'

    selected_states = st.multiselect(
        "Select states to compare:",
        sorted(responses_df['State Name'].dropna().unique()),
        default=["California", "Texas"]
    )

    state_trends = responses_df[responses_df['State Name'].isin(selected_states)]
    state_trends = state_trends.groupby(['Year', 'State Name'])['Response Rate (%)'].mean().reset_index()
    state_trends.rename(columns={'State Name': 'Region/State'}, inplace=True)
    combined_trends = pd.concat([national_trend, state_trends], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=combined_trends, x='Year', y='Response Rate (%)', hue='Region/State', marker='o', ax=ax)
    ax.set_title("Response Rate Trends")
    ax.grid(True)
    st.pyplot(fig)

    joined = pd.merge(state_results_df, responses_df, on=['Release Period', 'State'], how='left')
    corr_val = joined['Top-box Percentage'].corr(joined['Response Rate (%)'])
    st.metric(label="Correlation (Top-box % vs Response Rate)", value=f"{corr_val:.2f}")

# Tab 5: Opportunity Matrix
with tabs[5]:
    st.subheader("🧭 Opportunity Matrix")
    if 'Improvement' in pivot.columns:
        pivot['Latest Score'] = pivot[pivot.columns[-2]]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=pivot, x='Latest Score', y='Improvement', ax=ax)
        ax.axvline(pivot['Latest Score'].median(), color='red', linestyle='--')
        ax.axhline(pivot['Improvement'].median(), color='blue', linestyle='--')
        ax.set_title("Opportunity Matrix (Latest Score vs Improvement)")
        st.pyplot(fig)
    else:
        st.info("Opportunity matrix requires multi-year data.")

# Tab 6: AI Recommendations
with tabs[6]:
    st.subheader("💡 AI Recommendations")
    insights = []
    if 'Improvement' in pivot.columns:
        worst_declines = pivot.sort_values('Improvement').head(5).reset_index()
        for _, row in worst_declines.iterrows():
            q = row['Question'].lower()
            if 'discharge' in q:
                insights.append("🔍 Improve discharge communication procedures.")
            elif 'call' in q or 'help' in q:
                insights.append("🔍 Improve staff responsiveness to patient calls.")
            elif 'medicine' in q:
                insights.append("🔍 Clarify medication instructions.")
        if insights:
            for i in set(insights):
                st.markdown(f"- {i}")
        else:
            st.success("No critical declines detected.")

# Tab 7: State-Level Comparison
with tabs[7]:
    st.subheader("🏛️ Compare State vs National Scores")
    selected_state = st.selectbox("Select a State", sorted(state_results_df['State Name'].dropna().unique()))
    state_data = state_results_df[state_results_df['State Name'] == selected_state]
    natl_avg = national_results_df.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index()
    state_avg = state_data.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index(name='State Score')
    merged = pd.merge(state_avg, natl_avg, on=['Measure', 'Year'])

    compare_year = st.slider("Select Year", int(merged['Year'].min()), int(merged['Year'].max()), int(merged['Year'].max()), key='state_comparison_slider')
    year_df = merged[merged['Year'] == compare_year]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=year_df, x='Top-box Percentage', y='State Score', hue='Measure', s=80, ax=ax)
    ax.plot([0, 100], [0, 100], linestyle='--', color='gray')
    ax.set_xlabel("National Average")
    ax.set_ylabel(f"{selected_state} Average")
    ax.set_title(f"{selected_state} vs National Averages ({compare_year})")
    st.pyplot(fig)

    st.dataframe(year_df[['Measure', 'State Score', 'Top-box Percentage']])

# Tab 8: Patient Experience Heatmap
with tabs[8]:
    st.subheader("🗺️ Patient Experience Score by State")
    selected_year = st.slider("Select Year", int(state_results_df['Year'].min()), int(state_results_df['Year'].max()), int(state_results_df['Year'].max()), key="heatmap_year")
    selected_measure = st.selectbox("Select Measure", sorted(state_results_df['Measure'].dropna().unique()), key="heatmap_measure")

    filtered_df = state_results_df[
        (state_results_df['Year'] == selected_year) &
        (state_results_df['Measure'] == selected_measure)
    ]
    national_avg = filtered_df['Top-box Percentage'].mean()
    state_avg = filtered_df.groupby('State')['Top-box Percentage'].mean().reset_index()

    st.metric(label=f"National Avg – {selected_measure} ({selected_year})", value=f"{national_avg:.1f}%")

    try:
        fig = px.choropleth(
            state_avg,
            locations="State",
            locationmode="USA-states",
            scope="usa",
            color="Top-box Percentage",
            color_continuous_scale="RdYlGn",
            title=f"{selected_measure} – Top-box % by State ({selected_year})",
            labels={"Top-box Percentage": "Top-box %"}
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Plotly failed to render the heatmap.")
        st.error(e)

# Tab 9: Benchmarking Dashboard
with tabs[9]:
    st.subheader("🏅 State Benchmarking by Measure")
    selected_year = st.selectbox("Benchmarking Year", sorted(state_results_df['Year'].unique(), reverse=True), key='benchmark_year')
    selected_measure = st.selectbox("Benchmarking Measure", sorted(state_results_df['Measure'].dropna().unique()), key='benchmark_measure')

    bench_df = state_results_df[
        (state_results_df['Year'] == selected_year) &
        (state_results_df['Measure'] == selected_measure)
    ].groupby('State Name')['Top-box Percentage'].mean().reset_index()
    bench_df['Rank'] = bench_df['Top-box Percentage'].rank(ascending=False)
    bench_df = bench_df.sort_values('Top-box Percentage', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=bench_df, y='State Name', x='Top-box Percentage', palette='coolwarm', ax=ax)
    ax.set_title(f"State Benchmarking: {selected_measure} ({selected_year})")
    st.pyplot(fig)
    st.dataframe(bench_df.reset_index(drop=True))

# Tab 10: Anomaly Alerts
with tabs[10]:
    st.subheader("🚨 Anomaly Detection: Sudden Score Drops")
    anomaly_df = state_results_df.dropna(subset=['Top-box Percentage', 'Measure', 'State Name'])
    anomaly_df = anomaly_df.groupby(['State Name', 'Measure', 'Year'])['Top-box Percentage'].mean().reset_index()
    anomaly_df['YoY Change'] = anomaly_df.groupby(['State Name', 'Measure'])['Top-box Percentage'].diff()
    anomaly_df['Z-Score'] = anomaly_df.groupby('Measure')['YoY Change'].transform(lambda x: (x - x.mean()) / x.std())
    flagged = anomaly_df[(anomaly_df['Z-Score'] < -2) & (anomaly_df['YoY Change'] < 0)]

    if flagged.empty:
        st.success("No significant anomalies detected.")
    else:
        st.warning(f"{len(flagged)} anomalies found with Z-score < -2 (significant drops)")
        st.dataframe(flagged[['State Name', 'Measure', 'Year', 'Top-box Percentage', 'YoY Change', 'Z-Score']])

        pivot_alert = flagged.pivot_table(index='State Name', columns='Measure', values='YoY Change', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_alert, cmap="coolwarm", center=0, annot=True, fmt=".1f", linewidths=0.5, ax=ax)
        ax.set_title("Heatmap of Anomalous Drops")
        st.pyplot(fig)

    st.markdown("### 🔍 Possible Triggers")
    st.markdown("- Staffing shortages or leadership changes\n- Service interruptions or consolidations\n- Survey fatigue or response bias")

