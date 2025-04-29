import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# --------------------------------
# Streamlit Setup
# --------------------------------
st.set_page_config(page_title="HCAHPS Self-Service Explorer", layout="wide")
st.title("📋 HCAHPS Self-Service Insights Dashboard")

sns.set(style="whitegrid")
plt.close('all')

# --------------------------------
# Step 1: Load and Merge Data
# --------------------------------
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

# --------------------------------
# Setup Tabs
# --------------------------------
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

# National Trends
# National Trends
with tabs[0]:
    st.subheader("📈 National Top-box % by Year and Measure")

    national_results_df['Year'] = national_results_df['Year'].astype(int)  # Ensure no decimals

    # Group by Measure and Year
    measure_year_trend = national_results_df.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index()

    # Plot with seaborn lineplot for legend support
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=measure_year_trend, x='Year', y='Top-box Percentage', hue='Measure', ax=ax)
    ax.set_title("National Trends by Measure")
    ax.legend(title='Measure', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig)
   

# Most Improved Areas
with tabs[1]:
    st.subheader("📊 Most Improved Questions")

    state_results_df['Year'] = state_results_df['Year'].astype(int)  # Ensure no decimals

    # Group by Measure, Question, and Year
    q_year = state_results_df.groupby(['Measure', 'Question', 'Year'])['Top-box Percentage'].mean().reset_index()
    pivot = q_year.pivot(index=['Measure', 'Question'], columns='Year', values='Top-box Percentage')

    if pivot.shape[1] >= 2:
        pivot['Improvement'] = pivot[pivot.columns[-1]] - pivot[pivot.columns[0]]
        improved = pivot.sort_values('Improvement', ascending=False).reset_index()
        top_improved = improved[['Measure', 'Question', 'Improvement']].head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_improved, y='Question', x='Improvement', hue='Measure', legend=False, palette='viridis', ax=ax)
        st.pyplot(fig)

        st.dataframe(top_improved)

   


# Score Disparities
with tabs[2]:
    st.subheader("📉 Most Declined Questions")
    if 'Improvement' in pivot.columns:
        declined = pivot[pivot['Improvement'] < 0].sort_values('Improvement').reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=declined.head(10), y='Question', x='Improvement', hue='Question', legend=False, palette='flare', ax=ax)
        st.pyplot(fig)
        st.dataframe(declined[['Question', 'Improvement']].head(10))

# Regional Differences
with tabs[3]:
    st.subheader("🗺️ Regional Average Scores")

    # Ensure Year is integer to match slider
    state_results_df['Year'] = state_results_df['Year'].astype(int)
    reg_avg = state_results_df.groupby(['Region', 'Year'])['Top-box Percentage'].mean().reset_index()
    reg_avg['Year'] = reg_avg['Year'].astype(int)  # Ensure Year is int

    year = st.slider("Select Year", int(reg_avg['Year'].min()), int(reg_avg['Year'].max()), int(reg_avg['Year'].max()), key='regional_slider')
    chart_df = reg_avg[reg_avg['Year'] == year]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=chart_df, x='Top-box Percentage', y='Region', hue='Region', legend=False, palette='crest', ax=ax)
    st.pyplot(fig)



# Response Rate Insights
with tabs[4]:
    st.subheader("📬 National & State Response Rate Trends Over Time")

    # Ensure Year is integer
    responses_df['Year'] = pd.to_numeric(responses_df['Year'], errors='coerce').astype('Int64')
    responses_df.dropna(subset=['Year'], inplace=True)
    responses_df['Year'] = responses_df['Year'].astype(int)

    # National average response rate trend
    national_trend = responses_df.groupby('Year')["Response Rate (%)"].mean().reset_index()
    national_trend['Region/State'] = 'National'

    # Select states for comparison
    selected_states = st.multiselect(
        "Select states to compare with national trend:",
        sorted(responses_df['State Name'].dropna().unique()),
        default=["California", "Texas"]
    )

    # State-level response rate trend
    state_trends = responses_df[responses_df['State Name'].isin(selected_states)]
    state_trends = state_trends.groupby(['Year', 'State Name'])['Response Rate (%)'].mean().reset_index()
    state_trends.rename(columns={'State Name': 'Region/State'}, inplace=True)

    # Combine national + state data
    combined_trends = pd.concat([national_trend, state_trends], ignore_index=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=combined_trends, x='Year', y='Response Rate (%)', hue='Region/State', marker='o', ax=ax)
    ax.set_title("Response Rate Trends: National vs Selected States")
    ax.grid(True)
    st.pyplot(fig)

    # Correlation between Top-box % and Response Rate
    joined = pd.merge(state_results_df, responses_df, on=['Release Period', 'State'], how='left')
    corr_df = joined.dropna(subset=['Top-box Percentage', 'Response Rate (%)'])
    corr_val = corr_df['Top-box Percentage'].corr(corr_df['Response Rate (%)'])

    st.metric(label="Correlation (Top-box % vs Response Rate)", value=round(corr_val, 2))



# Opportunity Matrix
with tabs[5]:
    st.subheader("🧭 Opportunity Matrix")
    if 'Improvement' in pivot.columns:
        pivot['Latest Score'] = pivot[pivot.columns[-2]]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=pivot, x='Latest Score', y='Improvement', ax=ax)
        ax.axvline(pivot['Latest Score'].median(), color='red', linestyle='--')
        ax.axhline(pivot['Improvement'].median(), color='blue', linestyle='--')
        st.pyplot(fig)

# AI Recommendations
with tabs[6]:
    st.subheader("💡 AI Recommendations")
    insights = []
    if 'Improvement' in pivot.columns:
        worst_declines = pivot.sort_values('Improvement').head(5).reset_index()
        for _, row in worst_declines.iterrows():
            q = row['Question'].lower()
            if 'discharge' in q:
                insights.append("Improve discharge communication procedures.")
            elif 'call' in q or 'help' in q:
                insights.append("Improve staff responsiveness to patient calls.")
            elif 'medicine' in q:
                insights.append("Clarify medication instructions.")
        if insights:
            for i in set(insights):
                st.markdown(f"- {i}")
        else:
            st.success("No critical declines detected.")

# State-Level Comparison
with tabs[7]:
    st.subheader("🏛️ Compare State vs National Scores")
    selected_state = st.selectbox("Select a State", sorted(state_results_df['State Name'].dropna().unique()))
    state_data = state_results_df[state_results_df['State Name'] == selected_state]
    # Ensure Year is integer in all relevant DataFrames
    state_results_df['Year'] = state_results_df['Year'].astype(int)
    national_results_df['Year'] = national_results_df['Year'].astype(int)
    natl_avg = national_results_df.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index()
    state_avg = state_data.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index(name='State Score')
    merged = pd.merge(state_avg, natl_avg, on=['Measure', 'Year'])
    
    compare_year = st.slider("Select Year", int(merged['Year'].min()), int(merged['Year'].max()), int(merged['Year'].max()), key='state_comparison_slider')
    year_df = merged[merged['Year'] == compare_year].sort_values('State Score')
    
    # Scatter plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=year_df, x='Top-box Percentage', y='State Score', hue='Measure', s=80, ax=ax)
    ax.plot([0, 100], [0, 100], linestyle='--', color='gray')
    ax.set_xlabel("National Average")
    ax.set_ylabel(f"{selected_state} Average")
    st.pyplot(fig)
    
    st.dataframe(year_df[['Measure', 'State Score', 'Top-box Percentage']])
    
    # 💬 Recommendations based on gaps
    st.markdown("### 🔍 Recommendations Based on Gaps")

    year_df['Delta'] = year_df['State Score'] - year_df['Top-box Percentage']
    underperforming = year_df[year_df['Delta'] < -2].sort_values('Delta')
    outperforming = year_df[year_df['Delta'] > 2].sort_values('Delta', ascending=False)

    if not underperforming.empty:
        st.markdown("**⚠️ Areas where the state underperforms the national average:**")
        for _, row in underperforming.iterrows():
            st.markdown(f"- **{row['Measure']}**: Improve from {row['State Score']:.1f}% to match national average of {row['Top-box Percentage']:.1f}%.")
    else:
        st.success("This state is performing close to or above national averages across all measures.")

    if not outperforming.empty:
        st.markdown("**✅ Measures where the state outperforms the national average:**")
        for _, row in outperforming.iterrows():
            st.markdown(f"- **{row['Measure']}**: Excellent performance at {row['State Score']:.1f}% vs national {row['Top-box Percentage']:.1f}%.")

    st.info("Use this analysis to prioritize quality improvement efforts by focusing on underperforming areas first.")
# Patient Experience Heatmap
with tabs[8]:
    st.subheader("🗺️ Patient Experience Score by State")

    state_results_df['Year'] = state_results_df['Year'].astype(int)
    latest_year = state_results_df['Year'].max()

    state_avg = (
        state_results_df[state_results_df['Year'] == latest_year]
        .groupby('State Name')['Top-box Percentage']
        .mean()
        .reset_index()
    )

    try:
        import plotly.express as px

        fig = px.choropleth(
            state_avg,
            locations="State Name",
            locationmode="USA-states",
            scope="usa",
            color="Top-box Percentage",
            color_continuous_scale="RdYlGn",
            title=f"Top-box % by State ({latest_year})",
            labels={"Top-box Percentage": "Top-box %"},
        )
        fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning("Plotly is not available or an error occurred.")
        st.error(e)

# Benchmarking Dashboard
with tabs[9]:
    st.subheader("🏅 State Benchmarking by Measure")

    selected_year = st.selectbox("Select Year", sorted(state_results_df['Year'].unique(), reverse=True))
    selected_measure = st.selectbox("Select Measure", sorted(state_results_df['Measure'].dropna().unique()))

    bench_df = state_results_df[
        (state_results_df['Year'] == selected_year) &
        (state_results_df['Measure'] == selected_measure)
    ].groupby('State Name')['Top-box Percentage'].mean().reset_index()

    bench_df['Rank'] = bench_df['Top-box Percentage'].rank(ascending=False)
    bench_df = bench_df.sort_values('Top-box Percentage', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=bench_df, y='State Name', x='Top-box Percentage', hue='State Name', legend=False, palette='coolwarm', ax=ax)
    ax.set_title(f"Benchmarking: {selected_measure} ({selected_year})")
    st.pyplot(fig)

    st.dataframe(bench_df.reset_index(drop=True))

# Anomaly Alerts
with tabs[10]:
    st.subheader("🚨 Anomaly Detection: Sudden Score Drops")

    anomaly_df = state_results_df.copy()
    anomaly_df = anomaly_df.dropna(subset=['Top-box Percentage', 'Measure', 'State Name'])
    anomaly_df['Year'] = anomaly_df['Year'].astype(int)

    anomaly_df = anomaly_df.groupby(['State Name', 'Measure', 'Year'])['Top-box Percentage'].mean().reset_index()
    anomaly_df.sort_values(['State Name', 'Measure', 'Year'], inplace=True)
    anomaly_df['YoY Change'] = anomaly_df.groupby(['State Name', 'Measure'])['Top-box Percentage'].diff()

    anomaly_df['Z-Score'] = anomaly_df.groupby('Measure')['YoY Change'].transform(lambda x: (x - x.mean()) / x.std())

    flagged = anomaly_df[(anomaly_df['Z-Score'] < -2) & (anomaly_df['YoY Change'] < 0)]

    if flagged.empty:
        st.success("No significant anomalies detected based on year-over-year drops.")
    else:
        st.warning(f"{len(flagged)} anomalies found with Z-score < -2 (significant drops)")

        st.dataframe(flagged[['State Name', 'Measure', 'Year', 'Top-box Percentage', 'YoY Change', 'Z-Score']])

        st.markdown("### 📉 Heatmap of Anomalous Drops")

        pivot_alert = flagged.pivot_table(
            index='State Name', columns='Measure', values='YoY Change', aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_alert, cmap="coolwarm", center=0, annot=True, fmt=".1f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.markdown("### 🔍 Potential Triggers or Considerations")
        st.markdown("- Staffing shortages or policy changes")
        st.markdown("- Service disruption or hospital consolidation")
        st.markdown("- Low response rate or survey fatigue")
