import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)

# Using langchain_classic as strictly requested
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Config ---
st.set_page_config(page_title="Agentic RAG", layout="wide")

st.markdown(
    """
<style>
   .stChatInput {border-radius: 15px;}
   .stChatMessage {border-radius: 10px; margin-bottom: 10px;}
   .stSpinner {color: #00ff00;}
</style>
""",
    unsafe_allow_html=True,
)

# --- Session State Management ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


def clear_chat():
    """Callback to clear chat history securely before rerun."""
    st.session_state.chat_history = []


# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Agentic RAG System")

    model_options = {
        "Balanced (Recommended)": "Qwen/Qwen2.5-7B-Instruct",
        "Deep Thinker (Slower)": "deepseek-ai/deepseek-llm-7b-chat",
        "Fast & Lightweight": "microsoft/Phi-3.5-mini-instruct",
    }

    selected_assistant = st.selectbox(
        "Select AI Assistant:",
        options=list(model_options.keys()),
        index=0,
    )
    model_choice = model_options[selected_assistant]

    # Error handling for secrets to prevent crash if not set
    try:
        api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except KeyError:
        st.error("Missing HUGGINGFACEHUB_API_TOKEN in st.secrets")
        st.stop()

    st.divider()

    # File Upload Section
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)

    # --- Feature: Priority Selection ---
    priority_doc = None
    if uploaded_files:
        doc_names = [f.name for f in uploaded_files]
        priority_doc = st.selectbox(
            "Prioritize Document (Read First):", options=doc_names, index=0
        )

    process = st.sidebar.button("🚀 Process")

    # --- Feature: Clear Chat ---
    st.sidebar.button("🗑️ Clear Chat", on_click=clear_chat)

    st.divider()
    if st.session_state.vectorstore:
        st.success("Knowledge Base: Active ✅")
    else:
        st.warning("Knowledge Base: Empty ❌")

# --- Core Logic ---


@st.cache_data
def get_pdf_text(pdf_docs, priority_filename=None):
    """
    Extract text from PDFs with priority sorting.
    The priority file is moved to the front of the list to be processed first.
    """
    text = ""

    # Sort docs: Priority file comes first
    if priority_filename:
        pdf_docs = sorted(
            pdf_docs, key=lambda x: x.name == priority_filename, reverse=True
        )

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_vector_store(text):
    """Create Vector DB (Runs Locally - Free)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Local CPU Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_rag_chain(vectorstore, repo_id, hf_token):
    """
    Creates a Modern RAG Chain using LCEL.
    """
    # 1. Setup LLM
    # CRITICAL FIX: Added task="conversational"
    # This prevents the 'text-generation' ValueError for Instruct models.
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
        temperature=0.3,
        max_new_tokens=512,
        task="conversational",
    )

    chat_model = llm
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Final RAG Chain
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# --- Main App Interface ---

if len(st.session_state.chat_history) == 0:
    st.title("📄 Chat with your PDF")
    st.markdown("""
    Welcome! This app lets you upload a PDF and ask questions about it.
    
    **How to use:**
    1. **Sidebar (Left):** Upload a PDF document and click **"🚀 Process"**.
    2. **Below:** Type your question in the chat bar!
    """)

# Processing Logic
if process:
    if not uploaded_files:
        st.error("Please upload a PDF.")
    else:
        with st.spinner("Processing documents..."):
            # Pass the selected priority file to the extractor
            raw_text = get_pdf_text(uploaded_files, priority_filename=priority_doc)
            st.session_state.vectorstore = get_vector_store(raw_text)
            st.success("Ready! Chat below.")

# Chat Logic
user_query = st.chat_input("Ask a question...")

if user_query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first.")
    else:
        # Display User Message
        with st.chat_message("user"):
            st.write(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_chain = get_rag_chain(
                    st.session_state.vectorstore, model_choice, api_token
                )

                # Retrieve Chat History
                response = rag_chain.invoke(
                    {"input": user_query, "chat_history": st.session_state.chat_history}
                )
                answer = response["answer"]
                st.write(answer)

            # Update History
            st.session_state.chat_history.extend(
                [HumanMessage(content=user_query), AIMessage(content=answer)]
            )
