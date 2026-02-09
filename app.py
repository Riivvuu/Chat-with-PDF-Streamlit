import streamlit as st
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
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


def parse_response(text):
    """
    Extracts the 'Thinking' content and the 'Answer' content separately.
    Returns a tuple: (thinking_content, answer_content)
    """
    # Regex to find content inside <think> tags
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, text, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        # The answer is whatever's left after removing the <think> block
        answer = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()
    else:
        thinking = None
        answer = text.strip()

    answer = re.sub(r"\\\[", "$$", answer)
    answer = re.sub(r"\\\]", "$$", answer)
    answer = re.sub(r"\\\(", "$", answer)
    answer = re.sub(r"\\\)", "$", answer)

    return thinking, answer


# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Agentic RAG System")
    model_options = {
        "Balanced (Recommended)": "qwen/qwen3-32b",
        "Deep Thinker (Slower)": "openai/gpt-oss-120b",
        "Fast & Lightweight": "meta-llama/llama-4-scout-17b-16e-instruct",
    }

    def reset_conversation():
        st.session_state.chat_history = []

    selected_assistant = st.selectbox(
        "Select AI Assistant:",
        options=list(model_options.keys()),
        index=0,
        on_change=reset_conversation,
    )
    model_choice = model_options[selected_assistant]
    # Error handling for secrets to prevent crash if not set
    try:
        api_token = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("Missing GROQ_API_KEY in st.secrets")
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
        pdf_file.seek(0)  # Ensuring we're at the start of the file
        pdf_reader = PdfReader(pdf_file)
        file_text = ""
        for page in pdf_reader.pages:
            file_text += page.extract_text() or ""

        # Adding metadata so the LLM can "see" the source
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
def get_llm_chain(model_id, api_key):
    """
    Initialize Groq LLM with settings optimized for model type. Cached to prevent reloading on every chat message.
    """
    effort = None
    # Reasoning models need "room to breathe"
    if "gpt-oss" in model_id.lower() or "qwen3" in model_id.lower():
        effort = "high" if "gpt-oss" in model_id.lower() else "default"
        max_tokens = 8192  # Massive limit for thinking + answer
        temp = 0.6
    else:
        # Standard models work best with lower temp and standard limits
        temp = 0.3
        max_tokens = 1024  # Lower limit for faster, cost-efficient responses

    chat_model = ChatGroq(
        model_name=model_id,
        api_key=api_key,
        temperature=temp,
        max_tokens=max_tokens,
        reasoning_effort=effort,
    )
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
        "(mathematics, coding, legal, literature, science, history, etc.). "
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
        document_prompt=document_prompt,  # Passing custom prompt
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
            # Using fixed document loader
            raw_docs = get_pdf_documents(uploaded_files, priority_filename=priority_doc)
            st.session_state.vectorstore = get_vector_store(raw_docs)
            st.success("Ready! Chat below.")

# Displaying Chat History ---
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
        # Displaying User Message
        with st.chat_message("user"):
            st.write(user_query)

        # Generating Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Getting Cached LLM
                chat_model = get_llm_chain(model_choice, api_token)

                # Re-creating pipeline with current vectorstore
                # (We recreate the pipeline because vectorstore changes, but LLM is cached)
                priority_name = priority_doc if priority_doc else "None"
                rag_chain = create_rag_pipeline(
                    chat_model, st.session_state.vectorstore, priority_name
                )

                response = rag_chain.invoke(
                    {"input": user_query, "chat_history": st.session_state.chat_history}
                )
                raw_answer = response["answer"]
                # PARSING LOGIC: Separate thinking from the final answer
                thinking, final_answer = parse_response(raw_answer)

                # Visual Indicator: Verify which model was used
                st.caption(f"Generated by: {model_choice}")

                # If we found thinking content (DeepSeek behavior), show it in an expander
                if thinking:
                    with st.expander("🧠 Model  Thought Process"):
                        st.write(thinking)
                st.write(final_answer)

            st.session_state.chat_history.extend(
                [HumanMessage(content=user_query), AIMessage(content=final_answer)]
            )
