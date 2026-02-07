import streamlit as st
import re  # Added for cleaning <think> tags
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_core.documents import Document
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
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


def clean_response(text):
    """
    Removes <think>...</think> tags and their content from the text.
    Also removes any leading whitespace left behind.
    """
    # 1. Remove <think> tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. Fix LaTeX Delimiters for Streamlit
    # Replace block math \[ ... \] with $$ ... $$
    cleaned = re.sub(r"\\\[", "$$", text)
    cleaned = re.sub(r"\\\]", "$$", text)

    # Replace inline math \( ... \) with $ ... $
    cleaned = re.sub(r"\\\(", "$", text)
    cleaned = re.sub(r"\\\)", "$", text)

    return cleaned.strip()


# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Agentic RAG System")

    model_options = {
        "Balanced (Recommended)": "Qwen/Qwen2.5-7B-Instruct",
        "Deep Thinker (Slower)": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "Fast & Lightweight": "Qwen/Qwen2.5-1.5B-Instruct",
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
def get_pdf_documents(uploaded_files, priority_filename=None):
    """
    Extracts text and creates Document objects with metadata.
    This preserves file identity so the LLM knows which file is which.
    """
    documents = []

    # Sort files: Priority file first
    if priority_filename:
        uploaded_files = sorted(
            uploaded_files, key=lambda x: x.name == priority_filename, reverse=True
        )

    for pdf_file in uploaded_files:
        pdf_file.seek(0)  # Ensure we're at the start of the file
        pdf_reader = PdfReader(pdf_file)
        file_text = ""
        for page in pdf_reader.pages:
            file_text += page.extract_text() or ""

        # Add metadata so the LLM can "see" the source
        metadata = {
            "source": pdf_file.name,
            "is_priority": (pdf_file.name == priority_filename),
        }
        documents.append(Document(page_content=file_text, metadata=metadata))

    return documents


def get_vector_store(documents):
    """Create Vector DB (Runs Locally - Free)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Local CPU Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore


@st.cache_resource
def get_llm_chain(repo_id, hf_token):
    """
    Initialize LLM and Chain. Cached to prevent reloading on every chat message.
    """
    # 1. Initialize Endpoint
    # We explicitly set task="conversational" because Qwen 2.5 on the API
    # rejects the default 'text-generation' task check.
    # Adjust parameters based on model type
    if "deepseek" in repo_id.lower():
        # DeepSeek R1 requires higher temperature and massive token limit for thinking
        temp = 0.6
        max_tokens = 4096
    else:
        # Standard models (Qwen, Llama) work best with lower temp
        temp = 0.3
        max_tokens = 1024
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
        temperature=temp,
        max_new_tokens=max_tokens,
        task="conversational",
        return_full_text=False,
    )

    # 2. Wrap in Chat Interface
    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


def create_rag_pipeline(chat_model, vectorstore, priority_doc_name):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Contextualize Question Prompt
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

    # Answer Prompt
    qa_system_prompt = (
        "You are an intelligent assistant capable of analyzing documents from various domains "
        "(mathematics, coding, legal, literature, etc.). "
        "Use the following pieces of retrieved context to answer the question. "
        f"The user has prioritized the document: '{priority_doc_name}'. "
        "If you see information from that source, give it higher weight. "
        "If you don't know the answer, say that you don't know.\n\n"
        "**FORMATTING GUIDELINES:**\n"
        "1. **Math & Science:** Use standard LaTeX for all equations. "
        "   - Inline: $E=mc^2$ (single dollar signs)\n"
        "   - Block: $$x = ...$$ (double dollar signs)\n"
        "2. **Coding:** Always use triple backticks for code blocks with the language specified "
        "   (e.g., ```python for Python, ```sql for SQL).\n"
        "3. **General Structure:** Use bolding (**text**) for key terms and bullet points for lists.\n"
        "4. **Citations:** When answering based on specific documents, explicitly mention the source filename "
        "   (e.g., 'According to [Filename]...').\n"
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

    # Custom Document Prompt to inject metadata into the context string
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Source: {source}\nContent: {page_content}",
    )

    question_answer_chain = create_stuff_documents_chain(
        chat_model,
        qa_prompt,
        document_prompt=document_prompt,  # Pass custom prompt here
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# --- Main App Interface ---

if len(st.session_state.chat_history) == 0:
    st.title("📄 Chat with your PDF")
    st.markdown("""
    Welcome! This app lets you upload a PDF and ask questions about it.
    """)

# Processing Logic
if process:
    if not uploaded_files:
        st.error("Please upload a PDF.")
    else:
        with st.spinner("Processing documents..."):
            # Use the fixed document loader
            raw_docs = get_pdf_documents(uploaded_files, priority_filename=priority_doc)
            st.session_state.vectorstore = get_vector_store(raw_docs)
            st.success("Ready! Chat below.")

# Display Chat History ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

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
                # Get Cached LLM
                chat_model = get_llm_chain(model_choice, api_token)

                # Re-create pipeline with current vectorstore
                # (We recreate the pipeline because vectorstore changes, but LLM is cached)
                priority_name = priority_doc if priority_doc else "None"
                rag_chain = create_rag_pipeline(
                    chat_model, st.session_state.vectorstore, priority_name
                )

                response = rag_chain.invoke(
                    {"input": user_query, "chat_history": st.session_state.chat_history}
                )
                raw_answer = response["answer"]
                # CLEANING LOGIC 2: Clean new message before displaying
                final_answer = clean_response(raw_answer)
                st.write(final_answer)

            st.session_state.chat_history.extend(
                [HumanMessage(content=user_query), AIMessage(content=final_answer)]
            )
