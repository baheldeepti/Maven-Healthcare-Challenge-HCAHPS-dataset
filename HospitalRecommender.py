import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    "State-Level Comparison"
])

# National Trends
# National Trends
with tabs[0]:
    st.subheader("📈 National Average Top-box % by Year")
    national_results_df['Year'] = national_results_df['Year'].astype(int)  # Ensure no decimals
    national_avg = national_results_df.groupby('Year')["Top-box Percentage"].mean().reset_index()
    st.line_chart(national_avg.set_index("Year"))


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

        # 🧠 AI-style Recommendations
        st.markdown("### 💡 Recommendations for Continued Improvement")
        for _, row in top_improved.iterrows():
            measure = row['Measure'].lower()
            question = row['Question'].lower()

            if 'nurse' in question:
                rec = "Reinforce nurse communication protocols and bedside availability."
            elif 'doctor' in question:
                rec = "Sustain physician-patient communication clarity and trust-building."
            elif 'clean' in question or 'quiet' in question:
                rec = "Maintain hospital cleanliness and minimize nighttime disruptions."
            elif 'pain' in question:
                rec = "Continue effective pain management routines and check-ins."
            elif 'discharge' in question:
                rec = "Standardize discharge instruction practices across departments."
            elif 'call' in question or 'help' in question:
                rec = "Ensure timely response systems and increase staff awareness."

            else:
                rec = f"Maintain excellence in: {row['Measure']}"

            st.markdown(f"- **{row['Question']}** → {rec}")

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
    reg_avg = state_results_df.groupby(['Region', 'Year'])['Top-box Percentage'].mean().reset_index()
    year = st.slider("Select Year", int(reg_avg['Year'].min()), int(reg_avg['Year'].max()), int(reg_avg['Year'].max()), key='regional_slider')
    chart_df = reg_avg[reg_avg['Year'] == year]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=chart_df, x='Top-box Percentage', y='Region', hue='Region', legend=False, palette='crest', ax=ax)
    st.pyplot(fig)

# Response Rate Insights
with tabs[4]:
    st.subheader("📬 National Response Rate Over Time")
    rate_trend = responses_df.groupby('Year')["Response Rate (%)"].mean().reset_index()
    st.line_chart(rate_trend.set_index("Year"))
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
