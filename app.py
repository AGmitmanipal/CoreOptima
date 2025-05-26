# CoreOptima MVP: Predictive Lead Scoring and Funnel Insight Generator

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit as st
from openai import OpenAI

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

def visualize_funnel(df, mappings):
    funnel = df[mappings['Stage']].value_counts().reset_index()
    funnel.columns = ['Stage', 'Count']
    sns.barplot(x='Stage', y='Count', data=funnel)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

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
st.title("Purple Ash - Sales Funnel & Lead Conversion Optimizer")

uploaded_file = st.file_uploader("Upload your sales funnel CSV/XLSX file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = clean_data(df)
    mappings = map_columns(df)
    df = score_leads(df, mappings)

    st.subheader("🔍 Funnel Visualization")
    visualize_funnel(df, mappings)

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
