# CoreOptima MVP: Predictive Lead Scoring and Funnel Insight Generator


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="PurpleAsh - AI Business Optimization",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)



# Make sure this line appears before using TOGETHER_API_KEY
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]

# Create the Together.ai-compatible client
client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1"
)



MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def map_columns(df):
    st.sidebar.subheader("Map Your Columns")
    mappings = {}
    required_fields = {
        "Lead Name": None,
        "Stage": None,
        "Created Date": None,
        "Last Contact Date": None,
        "Status": None
    }

    for key in required_fields:
        options = df.columns.tolist()
        mappings[key] = st.sidebar.selectbox(f"Select column for '{key}'", options)

    return mappings

def clean_data(df):
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    return df

def score_leads(df, mappings):
    df = df.copy()
    df['days_since_contact'] = (datetime.now() - pd.to_datetime(df[mappings['Last Contact Date']], errors='coerce')).dt.days
    df['days_since_created'] = (datetime.now() - pd.to_datetime(df[mappings['Created Date']], errors='coerce')).dt.days

    # Rule-based scoring
    score = []
    for _, row in df.iterrows():
        s = 50
        if row['days_since_contact'] < 3:
            s += 25
        elif row['days_since_contact'] < 7:
            s += 10
        else:
            s -= 10
        if row['days_since_created'] > 30:
            s -= 5
        score.append(min(max(s, 0), 100))
    df['score'] = score
    return df


def v1(df, mappings):
    st.markdown("###🔍 Funnel Visualization")
    stage_counts = df[mappings['Stage']].value_counts()
    st.bar_chart(stage_counts)




def v2(df, mappings):
 st.markdown("### 🔔 Engagement vs Conversion Potential")
 df['engagement_bucket'] = pd.cut(df['days_since_contact'], bins=[-1, 3, 7, 30, 1000], 
                                 labels=["<3 Days", "3-7 Days", "7-30 Days", ">30 Days"])
 engaged_stats = df.groupby('engagement_bucket')['score'].mean().reset_index()
 st.write("📊 Avg. Lead Score by Engagement Recency")
 st.bar_chart(data=engaged_stats, x='engagement_bucket', y='score')

def v3(df, mappings):
 if 'Closed Won' in df[mappings['Status']].unique():
    st.markdown("### ⏱️ Time to Convert (Closed Won Only)")
    df['created_date'] = pd.to_datetime(df[mappings['Created Date']], errors='coerce')
    df['closed_date'] = pd.to_datetime(df[mappings['Last Contact Date']], errors='coerce')
    df['days_to_convert'] = (df['closed_date'] - df['created_date']).dt.days
    won_df = df[df[mappings['Status']] == "Closed Won"]
    st.write(f"Average days to convert: **{won_df['days_to_convert'].mean():.2f}**")
    st.line_chart(won_df['days_to_convert'].dropna())

def v4(df, mappings):
 if 'lead_source' in df.columns:
    st.markdown("### 🌍 Lead Source Conversion")
    df['is_won'] = df[mappings['Status']] == "Closed Won"
    conversion_by_source = df.groupby('lead_source')['is_won'].mean()
    st.bar_chart(conversion_by_source)

def v5(df, mappings):
 if 'sales_agent' in df.columns:
    st.markdown("### 👤 Sales Agent Success Rate")
    agent_stats = df.groupby('sales_agent')['score'].mean().sort_values(ascending=False)
    st.bar_chart(agent_stats)


def bucket_leads(df):
    top = df[df['score'] >= 80].head(5)
    mid = df[(df['score'] >= 50) & (df['score'] < 80)].head(10)
    low = df[df['score'] < 50].head(10)
    return top, mid, low

def generate_recommendations(top, mid, low):
    prompt = f"""
You are a B2B sales analyst. Based on this lead scoring:

Top Leads:
{top[['Lead Name', 'score']].to_string(index=False)}

Medium Leads:
{mid[['Lead Name', 'score']].to_string(index=False)}

Hard Leads:
{low[['Lead Name', 'score']].to_string(index=False)}

Give personalized, practical recommendations for how to convert each group of leads.
"""
    response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=512
)


    return response.choices[0].message.content


# Streamlit App
st.title("PurpleAsh - Sales Funnel & Lead Conversion Optimizer")

uploaded_file = st.file_uploader("Upload your sales funnel CSV/XLSX file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = clean_data(df)
    mappings = map_columns(df)
    df = score_leads(df, mappings)

    
    
    v1(df, mappings)
    v2(df, mappings)
    v3(df, mappings)
    v4(df, mappings)
    v5(df, mappings)

    st.subheader("📊 Lead Prioritization")
    top, mid, low = bucket_leads(df)
    st.write("✅ Top Leads:", top[['Lead Name', 'score']])
    st.write("⚖️ Medium Leads:", mid[['Lead Name', 'score']])
    st.write("⚠️ Hard Leads:", low[['Lead Name', 'score']])

    st.subheader("💡 Recommendations")
    with st.spinner("Generating insights..."):
        insights = generate_recommendations(top, mid, low)
        st.markdown(insights)

    st.download_button("Download Scored Leads", df.to_csv(index=False), file_name="scored_leads.csv")
