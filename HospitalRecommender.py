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
    trend_data = national_results_df.groupby(['Measure', 'Year'])['Top-box Percentage'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=trend_data, x='Year', y='Top-box Percentage', hue='Measure', ax=ax)
    ax.set_title("National Trends by Measure")
    ax.legend(title='Measure', bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)

    st.markdown("### 🤖 AI Summary (Beta)")
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
            with st.spinner("🔎 Analyzing trends with GPT-4..."):
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a healthcare analytics expert."},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0,
                    max_tokens=300
                )
                summary = response.choices[0].message.content
                st.markdown("### 📄 Executive Summary")
                st.write(summary)
        except Exception as e:
            st.error(f"⚠️ Error generating AI summary: {e}")

# Tab 1: Most Improved Areas with Composite Score and Bottom-box Trends
with tabs[1]:
    st.subheader("📊 Most Improved Patient Experience (Composite Score)")

    q_year = state_results_df.groupby(['Measure', 'Question', 'Year'])[['Top-box Percentage', 'Bottom-box Percentage']].mean().reset_index()
    q_year['Composite Score'] = q_year['Top-box Percentage'] - q_year['Bottom-box Percentage']
    pivot = q_year.pivot(index=['Measure', 'Question'], columns='Year', values='Composite Score')

    if pivot.shape[1] >= 2:
        pivot['Improvement'] = pivot[pivot.columns[-1]] - pivot[pivot.columns[0]]
        improved = pivot.reset_index()

        positive_improvement = improved[improved['Improvement'] > 0].sort_values('Improvement', ascending=False)
        weakest = improved.sort_values('Improvement').head(5)[['Measure', 'Question', 'Improvement']]
        top_improved = positive_improvement[['Measure', 'Question', 'Improvement']].head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_improved, y='Question', x='Improvement', hue='Measure', palette='crest', ax=ax, dodge=False)
        ax.set_title("Top 10 Questions by Composite Sentiment Improvement (Top-box % - Bottom-box %)")
        st.pyplot(fig)
        st.markdown("### 📈 Top Improved Composite Score Questions")
        st.dataframe(top_improved)
        st.markdown("### ⚠️ Weakest Performing Questions (Composite Score Decline)")
        st.dataframe(weakest)
        bottom_trend = state_results_df.groupby('Year')['Bottom-box Percentage'].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=bottom_trend, x='Year', y='Bottom-box Percentage', marker="o", ax=ax2)
        ax2.set_title("National Bottom-box % Trend Over Time")
        ax2.grid(True)
        st.pyplot(fig2)
        if not positive_improvement.empty:
            st.success("Great news! Several areas have shown positive improvements in patient experience.")
        else:
            st.warning("No positive composite score improvements found. Showing insights based on weak areas.")

        if st.checkbox("📄 Generate AI Summary & Recommendations (Composite + Bottom-box)"):
            bottom_trend = state_results_df.groupby('Year')['Bottom-box Percentage'].mean().reset_index()
            trend_text = bottom_trend.to_string(index=False)
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
{trend_text}
"""

            try:
                with st.spinner("Generating AI recommendations..."):
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a healthcare quality consultant."},
                            {"role": "user", "content": ai_prompt}
                        ],
                        temperature=0,
                        max_tokens=500
                    )
                    summary = response.choices[0].message.content
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
        # Step 1: Extract valid years dynamically
        year_cols = [col for col in pivot.columns if str(col).isdigit()]
        first_year, last_year = year_cols[0], year_cols[-1]

        # Step 2: Prepare full table with Measure and Question
        declined = pivot.reset_index().copy()  # FIXED: Reset index to bring Measure and Question as columns
        declined = declined[declined['Improvement'] < 0]
        declined = declined.sort_values('Improvement').reset_index(drop=True)
        top_declined = declined.head(10)

        # DEBUGGING aid: show available columns if error occurs
        if not all(col in top_declined.columns for col in ['Measure', 'Question', 'Improvement']):
            st.error("Missing required columns in `top_declined`. Columns found:")
            st.write(top_declined.columns.tolist())
        else:
            # Step 3: Visual - Declined barplot
            if not top_declined.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=top_declined,
                    y='Question',
                    x='Improvement',
                    hue='Measure',
                    palette='flare',
                    ax=ax,
                    dodge=False
                )
                ax.set_title("Top 10 Declined Questions by Composite Score")
                st.pyplot(fig)
            else:
                st.warning("⚠️ No declined questions found to visualize.")

            # Step 4: Table - Decline Details
            st.markdown("### 📋 Detailed Declines")
            st.dataframe(top_declined[['Measure', 'Question', 'Improvement']])

            # Step 5: Add start/end scores and conditional formatting
            declined['Start Score'] = declined[first_year]
            declined['End Score'] = declined[last_year]

            # Optional: Add visual indicators
            def format_change(val):
                if val < -5:
                    return f"🔻 {val:.2f}"
                elif val < 0:
                    return f"⬇️ {val:.2f}"
                else:
                    return f"{val:.2f}"

            styled_df = declined[['Measure', 'Question', 'Start Score', 'End Score', 'Improvement']].head(10).copy()
            styled_df['Improvement'] = styled_df['Improvement'].apply(format_change)

            st.markdown("### 📊 Percent Change with Trend Indicators")
            st.dataframe(
                styled_df.style
                .highlight_min(subset=['Improvement'], color='salmon', axis=0)
                .highlight_max(subset=['Improvement'], color='lightgreen', axis=0)
                .format({"Start Score": "{:.1f}", "End Score": "{:.1f}"})
            )
    else:
        st.warning("⚠️ 'Improvement' column not found in the dataset.")

      
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
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a healthcare data insights consultant."},
                        {"role": "user", "content": ai_prompt}
                    ],
                    temperature=0,
                    max_tokens=400
                )
                st.markdown("### 🤖 AI Summary & Recommendations")
                st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"⚠️ Error generating AI insights: {e}")
    

# Tab 4: Response Rate Insights
from openai import OpenAI

# instantiate once at the top of your script
client = OpenAI(api_key=openai.api_key)

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
    state_trends = (
        responses_df[responses_df['State Name'].isin(selected_states)]
        .groupby(['Year','State Name'])['Response Rate (%)']
        .mean()
        .reset_index()
    )
    state_trends.rename(columns={'State Name':'Region/State'}, inplace=True)

    # Combine all trends
    combined_trends = pd.concat([national_trend, state_trends], ignore_index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(
        data=combined_trends,
        x='Year', y='Response Rate (%)',
        hue='Region/State', marker='o',
        ax=ax
    )
    ax.set_title("📬 HCAHPS Survey Response Rate Trends")
    ax.grid(True)
    st.pyplot(fig)

    # Correlation merge
    joined = pd.merge(
        state_results_df,
        responses_df.drop(columns=['State Name'], errors='ignore'),
        on=['Release Period','State'],
        how='left'
    )
    national_corr = joined['Top-box Percentage'].corr(joined['Response Rate (%)'])
    st.metric(
        label="🧮 National Correlation: Top-box % vs Response Rate",
        value=f"{national_corr:.2f}"
    )

    # AI Summary
    if st.checkbox("📄 Generate AI Insights for Response Rate Correlation"):
        # Pivot for trend summary
        trend_summary = (
            combined_trends
            .pivot(index='Year', columns='Region/State', values='Response Rate (%)')
            .round(2)
        )

        # Compute per-state correlations
        state_corrs = []
        for state in selected_states:
            df_s = joined[joined['State Name'] == state]
            corr = df_s['Top-box Percentage'].corr(df_s['Response Rate (%)'])
            state_corrs.append((state, corr or 0.0))
        state_corr_text = "\n".join(f"{s}: {c:.2f}" for s,c in state_corrs)

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
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a healthcare survey data specialist."},
                        {"role":"user","content": ai_prompt}
                    ],
                    temperature=0,
                    max_tokens=500
                )
            st.markdown("### 🤖 AI Summary & Recommendations")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"⚠️ AI call failed: {e}")


# Tab 5: Opportunity Matrix
# Tab 5: Opportunity Matrix with AI Explanation & Insights
with tabs[5]:
    st.subheader("🧭 Opportunity Matrix")

    if 'Improvement' in pivot.columns:
        # Bring indices back into columns so we have Measure & Question fields
        omni = pivot.reset_index().copy()
        omni['Latest Score'] = omni[omni.columns[-2]]  # second-to-last column is last year’s composite

        # Plot scatter of Latest Score vs Improvement
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=omni,
            x='Latest Score',
            y='Improvement',
            ax=ax,
            s=100,
            edgecolor='w'
        )
        # median lines split into four quadrants
        med_x = omni['Latest Score'].median()
        med_y = omni['Improvement'].median()
        ax.axvline(med_x, color='red', linestyle='--', label=f"Median Score ({med_x:.1f})")
        ax.axhline(med_y, color='blue', linestyle='--', label=f"Median Improvement ({med_y:.1f})")
        ax.set_xlabel("Latest Composite Score (Top-box % − Bottom-box %)")
        ax.set_ylabel("Improvement (∆ Composite Score)")
        ax.set_title("Opportunity Matrix (Latest Score vs Improvement)")
        ax.legend(loc='lower right')
        st.pyplot(fig)

        # Explain the quadrants
        st.markdown("#### Quadrant Definitions")
        st.markdown("""
- **Top-Right** (High Score, High Improvement): Strengths to **maintain** and model elsewhere.  
- **Top-Left** (Low Score, High Improvement): **Rising stars**—invest to accelerate gains.  
- **Bottom-Right** (High Score, Low/Negative Improvement): **Stagnant leaders**—monitor for early decline.  
- **Bottom-Left** (Low Score, Low/Negative Improvement): **Critical laggards**—prioritize root-cause analysis here.
""")

        # AI-generated narrative & recommendations
        if st.checkbox("🤖 Generate AI Insights for Opportunity Matrix"):
            # prepare a concise list of top items in each quadrant
            tr = omni[(omni['Latest Score'] >= med_x) & (omni['Improvement'] >= med_y)]['Question'].tolist()
            tl = omni[(omni['Latest Score'] < med_x) & (omni['Improvement'] >= med_y)]['Question'].tolist()
            br = omni[(omni['Latest Score'] >= med_x) & (omni['Improvement'] < med_y)]['Question'].tolist()
            bl = omni[(omni['Latest Score'] < med_x) & (omni['Improvement'] < med_y)]['Question'].tolist()

            ai_prompt = f"""
You are a healthcare quality improvement consultant. Here is an opportunity matrix of HCAHPS questions, showing their most recent composite scores and improvements since the first year:

Rising Stars (Low Score, High Improvement): {tl}
Sustained Champions (High Score, High Improvement): {tr}
Potential Complacency (High Score, Low Improvement): {br}
Critical Laggards (Low Score, Low Improvement): {bl}

Please:
1. Summarize these four categories in 2–3 sentences each.
2. Provide 2–3 tailored recommendations for the Critical Laggards.
3. Suggest how to replicate Success Factors from the Sustained Champions in other areas.
"""

            try:
                with st.spinner("Generating AI-driven narrative…"):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert in healthcare patient experience analytics."},
                            {"role": "user",   "content": ai_prompt}
                        ],
                        temperature=0,
                        max_tokens=500
                    )
                st.markdown("### 🧠 AI-Generated Explanation & Insights")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"⚠️ AI Insights failed: {e}")
    else:
        st.info("Opportunity matrix requires multi-year data (an ‘Improvement’ column).")

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
# Tab 8: Patient Experience Heatmap + Leaderboard + AI Insights
with tabs[8]:
    st.subheader("🗺️ Patient Experience Score by State")

    # Year & measure selectors
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

    # Filter data
    filtered_df = state_results_df[
        (state_results_df['Year'] == selected_year) & 
        (state_results_df['Measure'] == selected_measure)
    ]
    national_avg = filtered_df['Top-box Percentage'].mean()
    state_avg = filtered_df.groupby('State')['Top-box Percentage'].mean().reset_index()

    st.metric(f"🇺🇸 National Avg – {selected_measure} ({selected_year})", f"{national_avg:.1f}%")

    # Heatmap
    try:
        fig = px.choropleth(
            state_avg, locations="State", locationmode="USA-states", scope="usa",
            color="Top-box Percentage", color_continuous_scale="RdYlGn",
            title=f"{selected_measure} – Top-box % by State ({selected_year})",
            labels={"Top-box Percentage":"Top-box %"}
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Plotly failed to render the heatmap.")
        st.error(e)

    # Leaderboard & bar chart
    st.subheader("🏆 State Leaderboard – Top-box %")
    leaderboard = state_avg.sort_values('Top-box Percentage', ascending=False).reset_index(drop=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 🟢 Top 10 States for '{selected_measure}'")
        st.dataframe(leaderboard.head(10).style.background_gradient(cmap='Greens', subset=['Top-box Percentage']))
    with col2:
        st.markdown(f"#### 🔴 Bottom 10 States for '{selected_measure}'")
        st.dataframe(leaderboard.tail(10).sort_values('Top-box Percentage').style.background_gradient(cmap='Reds_r', subset=['Top-box Percentage']))

    bar_fig, bar_ax = plt.subplots(figsize=(10,10))
    sns.barplot(data=leaderboard, x='Top-box Percentage', y='State', palette='Spectral', ax=bar_ax)
    bar_ax.set_title(f"{selected_measure} – State Rankings ({selected_year})")
    st.pyplot(bar_fig)

    csv = leaderboard.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Leaderboard CSV", data=csv,
                       file_name=f"leaderboard_{selected_measure}_{selected_year}.csv",
                       mime='text/csv')

    # AI Insights
    if st.checkbox("📄 Generate AI Summary for Top & Bottom States"):
        top_states = leaderboard.head(5).to_string(index=False)
        bottom_states = leaderboard.tail(5).sort_values('Top-box Percentage').to_string(index=False)

        ai_prompt = f"""
You are a healthcare quality consultant reviewing HCAHPS patient experience scores.
Measure: {selected_measure}
Year: {selected_year}

Top 5 States:
{top_states}

Bottom 5 States:
{bottom_states}

Please:
1. Identify and briefly describe standout (top) states.
2. Point out key challenges in the bottom states.
3. Provide 2–3 targeted, data-driven recommendations for improvement in the bottom group.
"""

        try:
            with st.spinner("Generating AI summary…"):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a healthcare performance improvement expert."},
                        {"role":"user",  "content": ai_prompt}
                    ],
                    temperature=0,
                    max_tokens=500
                )
            st.markdown("### 🤖 AI Insights & Recommendations")
            # ← use attribute access, not dict-style
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"⚠️ AI Insights failed: {e}")
    

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

