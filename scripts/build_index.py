# scripts/build_index.py
import os
import argparse
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

def build_search_index(ticker: str, pdf_path: str):
    # ✅ --- Foolproof path to the .env file ---
    # This finds the directory the script is in, goes one level up to the project root,
    # and then finds the .env file there. This works no matter where you run it from.
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    print(f"--- Starting indexing for {ticker.upper()} ---")

    # Explicitly load all credentials
    azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY")

    if not all([azure_openai_key, azure_openai_endpoint, azure_search_endpoint, azure_search_key]):
        raise ValueError("One or more required environment variables are missing from your .env file.")

    index_name = f"{ticker.lower()}-revenue-forecast-10k"
    
    # Load and chunk document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    # Initialize Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        api_key=azure_openai_key,
        azure_endpoint=azure_openai_endpoint
    )

    # Ingest data in batches
    batch_size = 50
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Ingesting batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")
        try:
            # The from_documents method can be used iteratively; it will add to the index if it already exists.
            AzureSearch.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=index_name,
                azure_search_endpoint=azure_search_endpoint,
                azure_search_key=azure_search_key,
            )
            time.sleep(1) # Wait between batches to respect free tier limits
        except Exception as e:
            print(f"   ---> ERROR on batch {i//batch_size + 1}: {e}")
            continue
    
    print(f"\n--- Indexing for {ticker.upper()} Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--file", type=str, required=True, help="Path to the 10-K PDF file relative to project root")
    args = parser.parse_args()
    
    build_search_index(args.ticker, args.file)