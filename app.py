import streamlit as st
import os
import pandas as pd
import xgboost as xgb
import json
import plotly.graph_objects as go

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Page Configuration & Theme ---
st.set_page_config(
    page_title="Quarterly Revenue Forecaster", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Improvement #2: New Modern Dark Theme
st.markdown("""
<style>
    /* Main background color */
    [data-testid="stAppViewContainer"] {
        background-color: #131720;
    }
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #202634;
    }
    /* Header/Title color */
    h1, h2, h3 {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Caching Functions for Performance ---
@st.cache_resource
def load_credentials():
    load_dotenv()
    return {
        "AZURE_OPENAI_ENDPOINT": st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
        "AZURE_OPENAI_API_KEY": st.secrets.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_SEARCH_ENDPOINT": st.secrets.get("AZURE_SEARCH_ENDPOINT", os.getenv("AZURE_SEARCH_ENDPOINT")),
        "AZURE_SEARCH_ADMIN_KEY": st.secrets.get("AZURE_SEARCH_ADMIN_KEY", os.getenv("AZURE_SEARCH_ADMIN_KEY"))
    }

@st.cache_resource
def load_models():
    model_lower = xgb.XGBRegressor(); model_lower.load_model("models/model_lower.json")
    model_median = xgb.XGBRegressor(); model_median.load_model("models/model_median.json")
    model_upper = xgb.XGBRegressor(); model_upper.load_model("models/model_upper.json")
    return model_lower, model_median, model_upper

@st.cache_data
def load_data():
    historical_df = pd.read_csv("data/historical_revenues.csv", parse_dates=['prediction_quarter'])
    forecast_features_df = pd.read_csv("data/features_for_forecast.csv")
    return historical_df, forecast_features_df

@st.cache_resource
def initialize_rag_components():
    llm = AzureChatOpenAI(azure_deployment="gpt-35-turbo", openai_api_version="2023-05-15")
    embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", openai_api_version="2023-05-15")
    template = """You are an expert financial analyst AI assistant...
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:"""
    prompt = ChatPromptTemplate.from_template(template)
    return llm, embeddings, prompt

# --- 3. Load All Assets ---
creds = load_credentials()
model_lower, model_median, model_upper = load_models()
historical_df, forecast_features_df = load_data()
llm, embeddings, prompt = initialize_rag_components()

# --- 4. Define Constants ---
MODEL_COLUMNS = ['revenues_lag_1', 'revenues_lag_2', 'revenues_lag_3', 'revenues_lag_4', 'revenues_rolling_avg_4', 'net_income_lag_2', 'net_income_lag_3', 'net_income_lag_4', 'net_income_rolling_avg_4', 'research_and_development_expense_lag_4', 'research_and_development_expense_lag_5', 'research_and_development_expense_lag_6', 'research_and_development_expense_lag_7', 'research_and_development_expense_lag_8', 'selling_general_and_administrative_expense_lag_1', 'selling_general_and_administrative_expense_lag_2', 'selling_general_and_administrative_expense_rolling_avg_4', 'assets_lag_2', 'assets_lag_3', 'assets_lag_4', 'liabilities_lag_2', 'liabilities_lag_3', 'liabilities_lag_4', 'shareholder_equity_lag_2', 'shareholder_equity_lag_3', 'shareholder_equity_lag_4', 'gdp_lag_1', 'gdp_lag_2', 'cpi_lag_1', 'unemployment_lag_1', 'unemployment_lag_2', 'year', 'quarter']


SHAP_FEATURE_MAP = {
    'revenues_lag_3': 'Revenue of 3 Quarters ago',
    'revenue_rolling_avg_4': 'Average Quarterly Revenue in the Last Year',
    'assets_lag_2': 'Assets of 2 Quarters ago',
    'net_income_lag_3': 'Net Income of 3 Quarters ago',
    'selling_general_and_administrative_expense_rolling_avg_4': 'Average Quarterly SG&A Expense in the Last Year'
}

# --- 5. UI Layout ---
st.title("📈 Quarterly Revenue Forecaster & AI Co-pilot")
st.sidebar.header("Select Company")
selected_ticker = st.sidebar.selectbox("Choose a stock ticker:", ("AAPL", "MSFT", "GOOGL"))

# --- 6. Forecasting Section ---
st.header(f"Next Quarter Revenue Forecast for {selected_ticker}")

if st.sidebar.button("Generate Forecast", type="primary"):
    with st.spinner("Generating forecast features..."):
        X_future = forecast_features_df[forecast_features_df['ticker'] == selected_ticker].copy()
        X_future['quarter'] = pd.Categorical(X_future['quarter'], categories=[1, 2, 3, 4], ordered=True)
        X_future = X_future[MODEL_COLUMNS]
    
    with st.spinner("Generating forecast..."):
        median = model_median.predict(X_future)[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            # ✅ Improvement #2: Simplified metric display
            st.metric(label="Predicted Revenue", value=f"${median/1e9:.2f} B")
            
            with st.expander("View Key Forecast Drivers"):
                st.write("Top 5 most influential factors for this forecast:")
                for feature, friendly_name in SHAP_FEATURE_MAP.items():
                    st.markdown(f"- **{friendly_name}**")

        with col2:
            history = historical_df[historical_df['ticker'] == selected_ticker].copy()
            last_historical_point = history.iloc[-1:]
            future_quarter_date = last_historical_point['prediction_quarter'].iloc[0] + pd.DateOffset(months=3)
            forecast_point = pd.DataFrame([{'prediction_quarter': future_quarter_date, 'revenues': median}])
            
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(x=history['prediction_quarter'], y=history['revenues'], mode='lines+markers', name='Historical Revenue'))
            fig_history.add_trace(go.Scatter(x=pd.concat([last_historical_point['prediction_quarter'], forecast_point['prediction_quarter']]), y=pd.concat([last_historical_point['revenues'], forecast_point['revenues']]), mode='lines', name='Forecast', line=dict(color='orange', dash='dash')))
            fig_history.add_trace(go.Scatter(x=forecast_point['prediction_quarter'], y=forecast_point['revenues'], mode='markers', marker_size=10, marker_color='orange', name='Forecast Point'))
            fig_history.update_layout(title=f"Historical vs. Forecasted Revenue", xaxis_title="Quarter", yaxis_title="Revenue (USD)", template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_history, use_container_width=True)

st.divider()

# --- 7. AI Co-pilot Section ---
st.header(f"Ask AI Co-pilot about {selected_ticker}'s 10-K Report")

# ✅ Improvement #3: Robust session state initialization for chat history
if "messages" not in st.session_state:
    st.session_state.messages = {} # Use a dictionary to store history per ticker

def run_rag_chain(question, ticker):
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                index_name = f"{ticker.lower()}-revenue-forecast-10k"
                vector_store = AzureSearch(azure_search_endpoint=creds["AZURE_SEARCH_ENDPOINT"], azure_search_key=creds["AZURE_SEARCH_ADMIN_KEY"], index_name=index_name, embedding_function=embeddings.embed_query)
                retriever = vector_store.as_retriever()
                rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                response = rag_chain.invoke(question)
                st.markdown(response)
                # Append to the correct ticker's history
                st.session_state.messages.setdefault(ticker, []).append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Could not retrieve answer. Please ensure the index for {ticker} has been built. Error: {e}")

# ✅ Improvement #3: Example questions
example_questions = ["What are the main business risks?", "Summarize Management's Discussion.", "Are there any ongoing legal proceedings?"]
cols = st.columns(len(example_questions))
for i, question in enumerate(example_questions):
    if cols[i].button(question, use_container_width=True, key=f"example_{i}"):
        # On button click, add user message and run the chain
        st.session_state.messages.setdefault(selected_ticker, []).append({"role": "user", "content": question})
        run_rag_chain(question, selected_ticker)

# Display chat history for the currently selected ticker
if selected_ticker in st.session_state.messages:
    for message in st.session_state.messages[selected_ticker]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Main chat input
if prompt_input := st.chat_input("Ask your own question..."):
    st.session_state.messages.setdefault(selected_ticker, []).append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)
    run_rag_chain(prompt_input, selected_ticker)