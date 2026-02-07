**Live demo:**
https://chat-with-pdf-app-nbe2gmyd8cf9pcsastsxqx.streamlit.app/

# 🤖 Agentic RAG System: Intelligent PDF Chatbot

An advanced, agent-driven Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, and Hugging Face. This application allows users to upload multiple PDF documents and engage in a context-aware conversation with specialized AI models, including the reasoning-heavy DeepSeek-R1.

## Key Features

    Multi-Model Selection: Choose between high-speed models (Qwen 2.5) and deep-reasoning models (DeepSeek-R1).
    Agentic Retrieval: Uses a "History-Aware Retriever" that rephrases user queries based on conversation context to ensure the AI always finds the most relevant information.
    Reasoning Transparency: Specifically designed to parse and display DeepSeek-R1's internal "thought process" in a dedicated UI expander.
    Document Prioritization: Select a "Primary Document" to give higher weight to specific sources during the search process.
    Scientific & Technical Ready: Full support for LaTeX mathematical rendering and syntax-highlighted code blocks.
    Local Embeddings: Efficient document indexing using sentence-transformers running locally on your CPU for better privacy and lower latency.

## Technology Stack

    Frontend: Streamlit
    Orchestration: LangChain (v1.0 patterns)
    Vector Database: FAISS (Facebook AI Similarity Search)
    Embeddings: all-MiniLM-L6-v2 (Local)
    LLM Hosting: Hugging Face Inference Endpoints

## Getting Started
1. Prerequisites

    Python 3.9+
    A Hugging Face Account and API Token (with 'Read' access).

2. Installation

Clone the repository and install the dependencies:

**Bash**
    `git clone https://github.com/Riivvuu/Chat-with-PDF-Streamlit
    cd agentic-rag-system
    pip install -r requirements.txt
    `

3. Configuration

The app uses Streamlit's secret management. Create a folder named .streamlit in your root directory and add a secrets.toml file:

**Ini, TOML**
    `#.streamlit/secrets.toml
    HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token_here"`

4. Running the App

Launch the Streamlit server:


**Bash**
    `streamlit run app.py`

## How to Use

    Select Assistant: Choose your preferred model in the sidebar (Recommended: Balanced for general use, Deep Thinker for complex math/logic).
    Upload PDFs: Drag and drop your documents into the sidebar.
    Prioritize: (Optional) Use the dropdown to select a document you want the AI to focus on primarily.
    Process: Click "🚀 Process" to index your files.
    Chat: Start asking questions! The AI will cite its sources and show its reasoning if using a DeepSeek model.

## Project Structure

    app.py: The main application logic and UI.
    requirements.txt: List of Python dependencies.
    .streamlit/secrets.toml: (Local only) Secure storage for API keys.
    README.md: Project documentation.

## License

Distributed under the MIT License. See LICENSE for more information.