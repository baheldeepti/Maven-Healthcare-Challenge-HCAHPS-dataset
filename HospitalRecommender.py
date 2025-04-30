# HospitalRecommender.py (fixed)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import openai
import os

# --------------------------------
# API Key Setup
# --------------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# --------------------------------
# Streamlit Setup
# --------------------------------
st.set_page_config(page_title="HCAHPS Self-Service Explorer", layout="wide")
st.title("HCAHPS Self-Service Insights Dashboard")
sns.set(style="whitegrid")
plt.close('all')

# --------------------------------
# Load and Merge Data
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

# Load CSVs
datasets = {name: pd.read_csv(github_base + filename) for name, filename in files.items()}
state_results_df = datasets['state_results']
national_results_df = datasets['national_results']
measures_df = datasets['measures']
questions_df = datasets['questions']
responses_df = datasets['responses']
states_df = datasets['states']

# Ensure correct types and merge info
responses_df['Response Rate (%)'] = pd.to_numeric(responses_df['Response Rate (%)'], errors='coerce')
for df in [state_results_df, national_results_df, responses_df]:
    df['Year'] = df['Release Period'].str[-4:].astype(int)

# Merge metadata
merged_measures_questions = pd.merge(measures_df, questions_df, on='Measure ID', how='left')
state_results_df = pd.merge(state_results_df, merged_measures_questions[['Measure ID', 'Measure', 'Question']], on='Measure ID', how='left')
national_results_df = pd.merge(national_results_df, merged_measures_questions[['Measure ID', 'Measure', 'Question']], on='Measure ID', how='left')

# Ensure 'State Name' exists by merging correctly
if 'State' in responses_df.columns and 'State' in states_df.columns:
    print
    responses_df = pd.merge(responses_df, states_df[['State', 'State Name', 'Region']], on='State', how='left')

if 'State' in state_results_df.columns and 'State' in states_df.columns:
    state_results_df = pd.merge(state_results_df, states_df[['State', 'State Name', 'Region']], on='State', how='left')

# Drop missing key values
state_results_df.dropna(subset=['Question', 'Top-box Percentage', 'Measure'], inplace=True)
national_results_df.dropna(subset=['Question', 'Top-box Percentage', 'Measure'], inplace=True)

joined = pd.merge(
    state_results_df,
    responses_df.drop(columns=['State Name'], errors='ignore'),
    on=['Release Period', 'State'],
    how='left',
    suffixes=('', '_resp')
)
# Check if 'State Name' exists, either from state_results_df or responses_df
if 'State Name' not in joined.columns and 'State Name_resp' in joined.columns:
    joined.rename(columns={'State Name_resp': 'State Name'}, inplace=True)

if 'State Name' not in joined.columns:
    st.error("❌ 'State Name' column is missing in merged data. Please check the dataset.")
else:
    st.success("✅ Data loaded and merged successfully with 'State Name' available.")


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

    # Calculate average Top-box % by year and measure
    measure_year_trend = national_results_df.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index()

    # Plot the trends
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=measure_year_trend, x='Year', y='Top-box Percentage', hue='Measure', ax=ax)
    ax.set_title("National Trends by Measure")
    ax.legend(title='Measure', bbox_to_anchor=(1.05, 1))
    ax.grid(True)
    st.pyplot(fig)

    # Optional AI Summary
   st.markdown("### \ud83e\udd16 AI Summary (Beta)")
    if st.checkbox("Generate AI Summary of National Trends"):
        trend_stats = (
            trend_data.groupby("Measure")['Top-box Percentage']
            .agg(['min', 'max', 'mean'])
            .reset_index()
            .round(2)
            .rename(columns={'min': 'Min %', 'max': 'Max %', 'mean': 'Mean %'})
        )

        st.dataframe(trend_stats)

        llm_prompt = f"""
        You are a healthcare analytics expert. Based on the following national HCAHPS measure summary (Top-box %), write an executive-level narrative.
        Identify key improvements, declines, and any surprising patterns.

        {trend_stats.to_string(index=False)}
        """

        try:
            with st.spinner("\ud83d\udd0e Analyzing trends with GPT-4..."):
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a healthcare analytics expert."},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0,
                    max_tokens=300
                )
                summary = response.choices[0].message.content
                st.markdown("### \ud83d\udcc4 Executive Summary")
                st.write(summary)
        except Exception as e:
            st.error(f"\u26a0\ufe0f Error generating AI summary: {e}")

# Tab 1: Most Improved Areas with Composite Score and Bottom-box Trends
with tabs[1]:
    st.subheader("📊 Most Improved Patient Experience (Composite Score)")

    # Calculate yearly Top and Bottom box % by question
    q_year = state_results_df.groupby(['Measure', 'Question', 'Year'])[
        ['Top-box Percentage', 'Bottom-box Percentage']
    ].mean().reset_index()

    # Calculate Composite Score = Top - Bottom
    q_year['Composite Score'] = q_year['Top-box Percentage'] - q_year['Bottom-box Percentage']

    # Pivot to compare first vs latest year
    pivot = q_year.pivot(index=['Measure', 'Question'], columns='Year', values='Composite Score')

    if pivot.shape[1] >= 2:
        pivot['Improvement'] = pivot[pivot.columns[-1]] - pivot[pivot.columns[0]]
        improved = pivot.reset_index()

        # Filter positive composite score improvement
        positive_improvement = improved[improved['Improvement'] > 0].sort_values('Improvement', ascending=False)
        weakest = improved.sort_values('Improvement').head(5)[['Measure', 'Question', 'Improvement']]

        if not positive_improvement.empty:
            top_improved = positive_improvement[['Measure', 'Question', 'Improvement']].head(10)

            # Plot top improved Composite Scores
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_improved, y='Question', x='Improvement', hue='Measure', palette='crest', ax=ax, dodge=False)
            ax.set_title("Top 10 Questions by Composite Sentiment Improvement (Top-box % - Bottom-box %)")
            st.pyplot(fig)

            st.markdown("### 📈 Top Improved Composite Score Questions")
            st.dataframe(top_improved)

            st.markdown("### ⚠️ Weakest Performing Questions (Composite Score Decline)")
            st.dataframe(weakest)

            # Optional: show Bottom-box trend over years
            st.subheader("📉 Bottom-box % Trend Over Time (Average by Year)")
            bottom_trend = state_results_df.groupby('Year')['Bottom-box Percentage'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=bottom_trend, x='Year', y='Bottom-box Percentage', marker="o", ax=ax2)
            ax2.set_title("National Bottom-box % Trend Over Time")
            ax2.grid(True)
            st.pyplot(fig2)
        else:
            st.warning("No positive composite score improvements found. Showing insights based on weak areas.")

        # AI Summary
        if st.checkbox("📄 Generate AI Summary & Recommendations (Composite + Bottom-box)"):
            ai_prompt = f"""
You are a healthcare data analyst. Based on the following data:
- Composite Score = Top-box % - Bottom-box %
- Bottom-box % trends nationally over time
Generate:
1. A short executive summary of the national trends.
2. Identify the top improved patient experience areas.
3. Highlight weak or declining areas based on composite score.
4. Provide 2–3 targeted recommendations to reduce bottom-box responses.

Top Improvements:
{top_improved.to_string(index=False) if not positive_improvement.empty else "None"}

Weakest Performing:
{weakest.to_string(index=False)}

Bottom-box Trend by Year:
{bottom_trend.to_string(index=False)}
"""

            try:
                with st.spinner("Generating AI recommendations..."):
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a healthcare quality consultant."},
                            {"role": "user", "content": ai_prompt}
                        ],
                        temperature=0,
                        max_tokens=500
                    )
                    summary = response.choices[0].message["content"]
                    st.markdown("### 🤖 AI Summary & Recommendations")
                    st.write(summary)
            except Exception as e:
                st.error(f"⚠️ Error generating AI summary: {e}")
    else:
        st.info("Insufficient data to calculate composite score improvement.")


# Tab 2: Score Disparities
with tabs[2]:
    st.subheader("📉 Most Declined Patient Experience Questions")

    if 'Improvement' in pivot.columns:
        # Filter most declined questions
        declined = pivot[pivot['Improvement'] < 0].sort_values('Improvement').reset_index()
        top_declined = declined.head(10)

        # Barplot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_declined, y='Question', x='Improvement', hue='Measure', palette='flare', ax=ax, dodge=False)
        ax.set_title("Top 10 Declined Questions by Composite Score")
        st.pyplot(fig)

        # Display table
        st.markdown("### 📋 Detailed Declines")
        st.dataframe(top_declined[['Measure', 'Question', 'Improvement']])

        # Show % Change Context
        st.markdown("### 📊 Percent Change (Contextual View)")
        first_year = pivot.columns[0]
        last_year = pivot.columns[-2]  # Improvement is last column
        declined[['Start Score', 'End Score']] = declined[[first_year, last_year]]
        st.dataframe(declined[['Measure', 'Question', 'Start Score', 'End Score', 'Improvement']].head(10))

        # AI Summary
        if st.checkbox("📄 Generate AI Insights for Declined Scores"):
            prompt = f"""
You are a healthcare analytics expert. The following HCAHPS questions have shown the greatest declines in composite patient experience scores (Top-box % - Bottom-box %) between {first_year} and {last_year}.
Provide:
1. An executive summary.
2. Hypotheses for why these areas might have declined.
3. Data-driven recommendations to reverse these trends.

Declined Questions:
{top_declined[['Measure', 'Question', 'Improvement']].to_string(index=False)}
"""

            try:
                with st.spinner("Analyzing declining scores with GPT-4..."):
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a healthcare improvement consultant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=500
                    )
                    ai_output = response.choices[0].message["content"]
                    st.markdown("### 🤖 AI Summary & Recommendations")
                    st.write(ai_output)
            except Exception as e:
                st.error(f"⚠️ Error generating AI insights: {e}")
    else:
        st.info("Insufficient year data to compute declines.")


# Tab 3: Regional Differences
with tabs[3]:
    st.subheader("🗺️ Regional Average Scores by Year")

    # Group data
    reg_avg = state_results_df.groupby(['Region', 'Year'])['Top-box Percentage'].mean().reset_index()

    # Year selector
    selected_year = st.slider("Select Year", int(reg_avg['Year'].min()), int(reg_avg['Year'].max()), int(reg_avg['Year'].max()), key='regional_year')
    chart_df = reg_avg[reg_avg['Year'] == selected_year].sort_values('Top-box Percentage', ascending=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=chart_df, x='Top-box Percentage', y='Region', palette='crest', ax=ax)
    ax.set_title(f"Regional HCAHPS Top-box % Scores – {selected_year}")
    st.pyplot(fig)

    # Top/Bottom performers
    top_region = chart_df.iloc[0]
    bottom_region = chart_df.iloc[-1]
    st.markdown("### 🏆 Top & Bottom Performing Regions")
    st.write(f"**Top Region:** {top_region['Region']} with {top_region['Top-box Percentage']:.2f}%")
    st.write(f"**Bottom Region:** {bottom_region['Region']} with {bottom_region['Top-box Percentage']:.2f}%")

    # Optional AI Insights
    if st.checkbox("📄 Generate AI Summary of Regional Differences"):
        ai_prompt = f"""
You are a healthcare analyst reviewing HCAHPS regional performance for {selected_year}.
Based on average Top-box percentages per region, summarize:
1. The highest and lowest scoring regions.
2. Possible factors driving the disparities (e.g., rural vs urban, staffing, access).
3. Regional strategies to improve patient experience in low-performing areas.

Data:
{chart_df[['Region', 'Top-box Percentage']].to_string(index=False)}
"""

        try:
            with st.spinner("Generating AI summary of regional differences..."):
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a healthcare data insights consultant."},
                        {"role": "user", "content": ai_prompt}
                    ],
                    temperature=0,
                    max_tokens=400
                )
                st.markdown("### 🤖 AI Summary & Recommendations")
                st.write(response.choices[0].message['content'])
        except Exception as e:
            st.error(f"⚠️ Error generating AI insights: {e}")


# Tab 4: Response Rate Insights
with tabs[4]:
    st.subheader("📬 National & State Response Rate Trends")

    # National trend
    national_trend = responses_df.groupby('Year')["Response Rate (%)"].mean().reset_index()
    national_trend['Region/State'] = 'National'

    # State selector
    selected_states = st.multiselect(
        "Select states to compare:",
        sorted(responses_df['State Name'].dropna().unique()),
        default=["California", "Texas"]
    )

    # State-level trends
    state_trends = responses_df[responses_df['State Name'].isin(selected_states)]
    state_trends = state_trends.groupby(['Year', 'State Name'])['Response Rate (%)'].mean().reset_index()
    state_trends.rename(columns={'State Name': 'Region/State'}, inplace=True)

    # Combine all trends
    combined_trends = pd.concat([national_trend, state_trends], ignore_index=True)

    # Line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=combined_trends, x='Year', y='Response Rate (%)', hue='Region/State', marker='o', ax=ax)
    ax.set_title("📬 HCAHPS Survey Response Rate Trends")
    ax.grid(True)
    st.pyplot(fig)

    # Merge for correlation analysis
    joined = pd.merge(state_results_df, responses_df, on=['Release Period', 'State'], how='left')

    # National correlation
    national_corr = joined['Top-box Percentage'].corr(joined['Response Rate (%)'])
    st.metric(label="🧮 National Correlation: Top-box % vs Response Rate", value=f"{national_corr:.2f}")


    # AI Summary
    if st.checkbox("📄 Generate AI Insights for Response Rate Correlation"):
        trend_summary = combined_trends.pivot(index='Year', columns='Region/State', values='Response Rate (%)').round(2)
        state_corr_text = "\n".join([f"{state}: {corr}" for state, corr in state_corrs])

        ai_prompt = f"""
You are a healthcare survey analytics expert. Based on:
- National correlation = {national_corr:.2f}
- State-level correlations:
{state_corr_text}

And response rate trends from {int(national_trend['Year'].min())} to {int(national_trend['Year'].max())}:

{trend_summary.to_string()}

Write:
1. A summary of how response rates correlate with satisfaction across states and nationally.
2. Any surprising findings.
3. 2–3 strategic suggestions to improve response rates and/or satisfaction levels.
"""

        try:
            with st.spinner("🔍 Generating AI-driven summary..."):
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a healthcare survey data specialist."},
                        {"role": "user", "content": ai_prompt}
                    ],
                    temperature=0,
                    max_tokens=500
                )
                summary = response.choices[0].message["content"]
                st.markdown("### 🤖 AI Summary & Recommendations")
                st.write(summary)
        except Exception as e:
            st.error(f"⚠️ Error generating AI summary: {e}")


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

# Tab 8: Patient Experience Heatmap + Leaderboard + Enhancements
with tabs[8]:
    st.subheader("🗺️ Patient Experience Score by State")

    # Year and Measure selectors
    selected_year = st.slider(
        "Select Year", 
        int(state_results_df['Year'].min()), 
        int(state_results_df['Year'].max()), 
        int(state_results_df['Year'].max()), 
        key="heatmap_year"
    )

    selected_measure = st.selectbox(
        "Select Measure", 
        sorted(state_results_df['Measure'].dropna().unique()), 
        key="heatmap_measure"
    )

    # Filter for selected year and measure
    filtered_df = state_results_df[
        (state_results_df['Year'] == selected_year) & 
        (state_results_df['Measure'] == selected_measure)
    ]
    national_avg = filtered_df['Top-box Percentage'].mean()
    state_avg = filtered_df.groupby('State')['Top-box Percentage'].mean().reset_index()

    st.metric(label=f"🇺🇸 National Avg – {selected_measure} ({selected_year})", value=f"{national_avg:.1f}%")

    # Plotly heatmap
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

    # Leaderboard
    st.subheader("🏆 State Leaderboard – Top-box %")
    leaderboard = state_avg.sort_values('Top-box Percentage', ascending=False).reset_index(drop=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 🟢 Top 10 States for '{selected_measure}'")
        st.dataframe(leaderboard.head(10).style.background_gradient(cmap='Greens', subset=['Top-box Percentage']))
    with col2:
        st.markdown(f"#### 🔴 Bottom 10 States for '{selected_measure}'")
        st.dataframe(leaderboard.tail(10).sort_values('Top-box Percentage').style.background_gradient(cmap='Reds_r', subset=['Top-box Percentage']))

    # Horizontal Bar Chart beside map
    st.subheader("📊 Ranked Scores by State")
    bar_fig, bar_ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=leaderboard, x='Top-box Percentage', y='State', palette='Spectral', ax=bar_ax)
    bar_ax.set_title(f"{selected_measure} – State Rankings ({selected_year})")
    st.pyplot(bar_fig)

    # Download leaderboard
    csv = leaderboard.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Leaderboard CSV", data=csv, file_name=f"leaderboard_{selected_measure}_{selected_year}.csv", mime='text/csv')

    # AI Summary
    if st.checkbox("📄 Generate AI Summary for Top & Bottom States"):
        top_states = leaderboard.head(5).to_string(index=False)
        bottom_states = leaderboard.tail(5).sort_values('Top-box Percentage').to_string(index=False)

        ai_prompt = f"""
You are a healthcare quality consultant reviewing HCAHPS patient experience scores.
The measure is: {selected_measure}
The year is: {selected_year}

Based on this, please:
1. Identify standout states (Top 5)
2. Call out low-performing states (Bottom 5)
3. Suggest 2–3 actions for states at the bottom of the list to improve.

Top States:
{top_states}

Bottom States:
{bottom_states}
"""

        try:
            with st.spinner("Generating AI summary..."):
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a healthcare performance improvement expert."},
                        {"role": "user", "content": ai_prompt}
                    ],
                    temperature=0,
                    max_tokens=500
                )
                summary = response.choices[0].message['content']
                st.markdown("### 🤖 AI Insights & Recommendations")
                st.write(summary)
        except Exception as e:
            st.error(f"⚠️ Error generating AI summary: {e}")

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

# Tab 10: Anomaly Alerts – Sudden Drops with Visual Enhancements
with tabs[10]:
    st.subheader("🚨 Anomaly Detection: Sudden Score Drops")

    # Prepare data
    anomaly_df = state_results_df.dropna(subset=['Top-box Percentage', 'Measure', 'State Name'])
    anomaly_df = anomaly_df.groupby(['State Name', 'Measure', 'Year'])['Top-box Percentage'].mean().reset_index()
    anomaly_df['YoY Change'] = anomaly_df.groupby(['State Name', 'Measure'])['Top-box Percentage'].diff()
    anomaly_df['Z-Score'] = anomaly_df.groupby('Measure')['YoY Change'].transform(lambda x: (x - x.mean()) / x.std())

    # Filter anomalies
    flagged = anomaly_df[(anomaly_df['Z-Score'] < -2) & (anomaly_df['YoY Change'] < 0)]

    if flagged.empty:
        st.success("✅ No significant anomalies detected. Patient experience scores are stable.")
    else:
        st.warning(f"⚠️ {len(flagged)} anomalies found with Z-score < -2 and YoY drop (potential concerns)")

        # Color-coded styled table
        styled_table = flagged[['State Name', 'Measure', 'Year', 'Top-box Percentage', 'YoY Change', 'Z-Score']].style\
            .background_gradient(subset=['YoY Change'], cmap='Reds_r')\
            .format({'Top-box Percentage': '{:.1f}%', 'YoY Change': '{:.1f}', 'Z-Score': '{:.2f}'})

        st.dataframe(styled_table)

        # Heatmap: State x Measure showing avg YoY drop
        pivot_alert = flagged.pivot_table(index='State Name', columns='Measure', values='YoY Change', aggfunc='mean')

        st.markdown("### 🔥 Heatmap of Anomalous Score Drops")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            pivot_alert, 
            cmap="coolwarm", 
            center=0, 
            annot=True, 
            fmt=".1f", 
            linewidths=0.5, 
            cbar_kws={'label': 'YoY Change'}
        )
        ax.set_title("Significant Negative YoY Changes (Z < -2)", fontsize=14, pad=12)
        st.pyplot(fig)

    # Triggers Section
    st.markdown("### 🔍 Possible Triggers for Anomalies")
    st.markdown("""
- 🧑‍⚕️ **Staffing shortages** or leadership turnover  
- ⚙️ **Operational disruptions** (e.g., closures, consolidations)  
- 📉 **Survey response fatigue** or demographic shifts  
- 📋 **Data capture errors** or missing survey periods  
    """)

