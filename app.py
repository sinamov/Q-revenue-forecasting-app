# app.py
import streamlit as st
import os
import pandas as pd
import xgboost as xgb
import json
import requests

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Quarterly Revenue Forecaster", layout="wide")
st.title("📈 Quarterly Revenue Forecaster & AI Co-pilot")

# --- Load Credentials (cached for performance) ---
@st.cache_resource
def load_credentials():
    load_dotenv()
    # Note: For Streamlit Cloud, you'll set these as Secrets, not in a .env file
    return {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "AZURE_SEARCH_ADMIN_KEY": os.getenv("AZURE_SEARCH_ADMIN_KEY")
    }
creds = load_credentials()

# --- Load Models (cached for performance) ---
@st.cache_resource
def load_models():
    """Loads the three trained XGBoost models from disk."""
    model_lower = xgb.XGBRegressor(); model_lower.load_model("models/model_lower.json")
    model_median = xgb.XGBRegressor(); model_median.load_model("models/model_median.json")
    model_upper = xgb.XGBRegressor(); model_upper.load_model("models/model_upper.json")
    return model_lower, model_median, model_upper
model_lower, model_median, model_upper = load_models()

# --- Initialize RAG Components (cached for performance) ---
@st.cache_resource
def initialize_rag_components():
    """Initializes the lightweight components of the RAG chain."""
    llm = AzureChatOpenAI(azure_deployment="gpt-35-turbo", openai_api_version="2023-05-15")
    embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", openai_api_version="2023-05-15")
    template = """
    You are an expert financial analyst AI assistant. Use the following pieces of retrieved context 
    from a company's 10-K report to answer the user's question. 
    If you don't know the answer from the context, just say that you don't know. 
    Provide a detailed and insightful answer based on the provided text.
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    return llm, embeddings, prompt
llm, embeddings, prompt = initialize_rag_components()

# --- UI Elements ---
st.sidebar.header("Select Company")
selected_ticker = st.sidebar.selectbox("Choose a stock ticker:", ("AAPL", "MSFT", "GOOGL"))

# --- FORECASTING SECTION ---
st.header(f"Next Quarter Revenue Forecast for {selected_ticker}")

# This data is the last row of features we generated in our notebook.
# In a real production system, you would generate this dynamically.
future_features_data = {
    "AAPL": [{"year": 2025, "quarter": 4, "revenues_lag_1": 94036000000, "revenues_lag_2": 95359000000, "revenues_lag_3": 124300000000, "revenues_lag_4": 94930000000, "revenues_rolling_avg_4": 102156250000.0, "net_income_lag_2": 23636000000, "net_income_lag_3": 33916000000, "net_income_lag_4": 22722000000, "net_income_rolling_avg_4": 26053000000.0, "research_and_development_expense_lag_4": 7905000000, "research_and_development_expense_lag_5": 7711000000, "research_and_development_expense_lag_6": 7439000000, "research_and_development_expense_lag_7": 7307000000, "research_and_development_expense_lag_8": 6795000000, "selling_general_and_administrative_expense_lag_1": 6220000000, "selling_general_and_administrative_expense_lag_2": 7239000000, "selling_general_and_administrative_expense_rolling_avg_4": 6867000000.0, "assets_lag_2": 352583000000, "assets_lag_3": 353589000000, "assets_lag_4": 335038000000, "liabilities_lag_2": 272990000000, "liabilities_lag_3": 274805000000, "liabilities_lag_4": 262334000000, "shareholder_equity_lag_2": 79593000000, "shareholder_equity_lag_3": 78784000000, "shareholder_equity_lag_4": 72704000000, "gdp_lag_1": 28679.8, "gdp_lag_2": 28362.8, "cpi_lag_1": 314.069, "unemployment_lag_1": 3.8, "unemployment_lag_2": 3.9, "revenue_seasonal_diff": -8120250000.0}],
    # NOTE: Add the feature rows for MSFT and GOOGL here from your notebook's X_future DataFrame
    "MSFT": [{"year": 2025, "quarter": 4, "revenues_lag_1": 76441000000, "revenues_lag_2": 70066000000, "revenues_lag_3": 69632000000, "revenues_lag_4": 65585000000, "revenues_rolling_avg_4": 70431000000.0, "net_income_lag_2": 21870000000, "net_income_lag_3": 21939000000, "net_income_lag_4": 20081000000, "net_income_rolling_avg_4": 22295000000.0, "research_and_development_expense_lag_4": 6848000000, "research_and_development_expense_lag_5": 7114000000, "research_and_development_expense_lag_6": 6846000000, "research_and_development_expense_lag_7": 6997000000, "research_and_development_expense_lag_8": 6766000000, "selling_general_and_administrative_expense_lag_1": 8475000000, "selling_general_and_administrative_expense_lag_2": 7717000000, "selling_general_and_administrative_expense_rolling_avg_4": 8328750000.0, "assets_lag_2": 484252000000, "assets_lag_3": 470605000000, "assets_lag_4": 427248000000, "liabilities_lag_2": 232049000000, "liabilities_lag_3": 224902000000, "liabilities_lag_4": 205739000000, "shareholder_equity_lag_2": 252203000000, "shareholder_equity_lag_3": 245703000000, "shareholder_equity_lag_4": 221509000000, "gdp_lag_1": 28679.8, "gdp_lag_2": 28362.8, "cpi_lag_1": 314.069, "unemployment_lag_1": 3.8, "unemployment_lag_2": 3.9, "revenue_seasonal_diff": 6010000000.0}],
    "GOOGL": [{"year": 2025, "quarter": 4, "revenues_lag_1": 96428000000, "revenues_lag_2": 90234000000, "revenues_lag_3": 96469000000, "revenues_lag_4": 88268000000, "revenues_rolling_avg_4": 92849750000.0, "net_income_lag_2": 15051000000, "net_income_lag_3": 19689000000, "net_income_lag_4": 16436000000, "net_income_rolling_avg_4": 18451250000.0, "research_and_development_expense_lag_4": 11370000000, "research_and_development_expense_lag_5": 11195000000, "research_and_development_expense_lag_6": 10519000000, "research_and_development_expense_lag_7": 10123000000, "research_and_development_expense_lag_8": 9789000000, "selling_general_and_administrative_expense_lag_1": 9568000000, "selling_general_and_administrative_expense_lag_2": 8205000000, "selling_general_and_administrative_expense_rolling_avg_4": 9477250000.0, "assets_lag_2": 407338000000, "assets_lag_3": 404391000000, "assets_lag_4": 389474000000, "liabilities_lag_2": 118434000000, "liabilities_lag_3": 118228000000, "liabilities_lag_4": 110996000000, "shareholder_equity_lag_2": 288904000000, "shareholder_equity_lag_3": 286163000000, "shareholder_equity_lag_4": 278478000000, "gdp_lag_1": 28679.8, "gdp_lag_2": 28362.8, "cpi_lag_1": 314.069, "unemployment_lag_1": 3.8, "unemployment_lag_2": 3.9, "revenue_seasonal_diff": 3578250000.0}]
}

if st.sidebar.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        # Prepare the feature DataFrame
        X_future = pd.DataFrame.from_dict(future_features_data[selected_ticker])
        X_future['quarter'] = pd.Categorical(X_future['quarter'], categories=[1, 2, 3, 4], ordered=True)
        
        # Make predictions
        lower = model_lower.predict(X_future)[0]
        median = model_median.predict(X_future)[0]
        upper = model_upper.predict(X_future)[0]
        
        # Post-process to prevent crossing
        final_lower = min(lower, median, upper)
        final_upper = max(lower, median, upper)

        # Display results
        st.metric(label="Predicted Revenue (Median)", value=f"${median/1e9:.2f} B")
        st.info(f"The 80% prediction interval is between ${final_lower/1e9:.2f} B and ${final_upper/1e9:.2f} B.")

st.divider()

# --- AI CO-PILOT SECTION ---
st.header(f"Ask AI Co-pilot about {selected_ticker}'s 10-K Report")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt_input := st.chat_input("What are the company's main business risks?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            index_name = f"{selected_ticker.lower()}-revenue-forecast-10k"
            
            # Initialize the retriever for the specific company
            vector_store = AzureSearch(
                azure_search_endpoint=creds["AZURE_SEARCH_ENDPOINT"],
                azure_search_key=creds["AZURE_SEARCH_ADMIN_KEY"],
                index_name=index_name,
                embedding_function=embeddings.embed_query,
            )
            retriever = vector_store.as_retriever()
            
            # Construct and invoke the RAG chain
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
            )
            response = rag_chain.invoke(prompt_input)
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})