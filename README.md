## **Live demo:**
### **https://chat-with-pdf-app-nbe2gmyd8cf9pcsastsxqx.streamlit.app/**

# 🤖 Agentic RAG System: 📄Chat with your PDF

An advanced, agent-driven Retrieval-Augmented Generation (Agentic RAG) system built with Streamlit, LangChain, and Hugging Face. This application allows users to upload multiple PDF documents and engage in a context-aware conversation with specialized AI models, including the reasoning-heavy DeepSeek-R1.

## Key Features

* Multi-Model Selection: Switch between Qwen 2.5 for fast responses and DeepSeek-R1 for deep reasoning tasks.
* Reasoning Transparency: View the AI's internal "thought process" (Chain-of-Thought) when using DeepSeek models.
* Agentic Retrieval: Uses a "History-Aware Retriever" that rephrases user queries based on conversation context to ensure the AI always finds the most relevant information.
* Source Citation: The assistant explicitly mentions which document it is referencing.
* Document Prioritization: Select a "Primary Document" to give higher weight to specific sources during the search process.
* Scientific & Technical Ready: Full support for LaTeX mathematical rendering and syntax-highlighted code blocks.
* Local Embeddings: Efficient document indexing using sentence-transformers running locally on your CPU for better privacy and lower latency.

## Technology Stack
* Frontend: Streamlit
* Orchestration: LangChain (v1.0 patterns)
* Vector Database: FAISS (Facebook AI Similarity Search)
* Embeddings: all-MiniLM-L6-v2 (Local)
* LLM Hosting: Hugging Face Inference Endpoints

## Local Setup
1. Prerequisites-
    Python 3.9+
    A Hugging Face Account and API Token (with 'Read' access).

2. Installation
    Clone the repository and install the dependencies:
    **Bash**

        git clone https://github.com/Riivvuu/Chat-with-PDF-Streamlit
        cd Chat-with-PDF-Streamlit
        pip install -r requirements.txt
        

3. Configuration
    Create a .streamlit/secrets.toml file in the project folder and add your Hugging Face API token:
    **Ini, TOML**

        HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token_here"

4. Running the App
    Launch the Streamlit server:
    **Bash**

        streamlit run app.py

## How it works

1. Ingestion: Documents are split into 1000-character chunks with a 200-character overlap to preserve context.
2. Vector Store: Chunks are converted into numerical embeddings using all-MiniLM-L6-v2 and stored in a FAISS index for lightning-fast search.
3. Agentic Loop: The system uses a "History-Aware Retriever" to refine user queries before searching, ensuring the AI finds the most relevant "slices" of information.

## Project Structure

    app.py: The main application logic and UI.
    requirements.txt: List of Python dependencies.
    .streamlit/secrets.toml: (Local only) Secure storage for API keys.
    README.md: Project documentation.

## License

Distributed under the MIT License. See LICENSE for more information.