# HCAHPS Self-Service Explorer 🏥

An interactive decision support dashboard built for the **Maven Healthcare Analytics Challenge**. This project empowers healthcare teams to explore **patient experience data** from HCAHPS surveys across U.S. states and regions.

## 📌 Features
- Explore **national and regional trends** in Top-box % over time
- Identify **most improved** and **declining questions** using composite scores
- Visualize disparities, anomalies, and benchmarks by state
- Understand the **correlation between response rate and satisfaction**
- Use the **Opportunity Matrix** to prioritize patient experience gaps
- Generate **executive summaries** powered by GPT-4 (OpenAI API)

## 🧠 AI-Powered Tabs
- GPT-4 generated summaries for national trends, bottom-box scores, regional gaps, and performance opportunities
- Real-time narrative generation for executives and stakeholders

## 📊 Sample Insights
- Discharge communication improved nationally, but staff responsiveness declined
- Midwest states reported the greatest gains in hospital quietness
- States like Texas showed high correlation between response rates and satisfaction

## 💻 Tech Stack
- Python, Streamlit, Pandas, Plotly, Seaborn
- GPT-4 via OpenAI API
- GitHub-hosted CSVs for reproducibility

## 🔗 Links
- [Live Streamlit App](https://hospital-bi-tool.streamlit.app)
- [GitHub Repository](https://github.com/baheldeepti/Maven-Healthcare-Challenge-HCAHPS-dataset)

## 📈 Metrics Explained
- **Top-box %**: % of patients rating 9–10 or “Always”
- **Composite Score**: Top-box % − Bottom-box %
- **OTP (On-Time Performance)**: proxy for responsiveness
- **CASM, RASM**: Not used here but relevant in health ops

## 📬 Author
**Deepti Bahel**  
Senior BI Engineer | Healthcare Analytics  
[LinkedIn Profile](https://www.linkedin.com/in/deeptibahel/)
