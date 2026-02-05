import streamlit as st
import pymupdf4llm
import tempfile
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get(
    "HUGGINGFACEHUB_API_TOKEN"
)
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
else:
    st.error("Missing HuggingFace Token. Check your Streamlit Secrets.")


def get_pdf_text(pdf_docs):
    text = ""
    pdf_docs = sorted(pdf_docs, key=lambda x: x.name)
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.getbuffer())
            tmp_path = tmp_file.name
        try:
            text += f"\n\n--- SOURCE DOCUMENT: {pdf.name} ---\n"
            text += pymupdf4llm.to_markdown(tmp_path)
        finally:
            os.remove(tmp_path)
    return text


def get_text_chunks(text):
    # Balanced chunk size: 700 chars is large enough for context, small enough for Zephyr
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store(text_chunks):
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    # FINAL PROMPT STRATEGY: "Guarded Generalist"
    # 1. Logic: Explicitly handles Code vs. Narrative.
    # 2. Safety: top_k=50 physically prevents "fake word" invention.

    prompt_template = """
    You are an intelligent document assistant. Analyze the provided context and summarize it according to its type.

    Context:
    {context}

    Question: 
    {question}

    ----------------
    UNIVERSAL INSTRUCTIONS:
    
    1. **Identify the Document Type:**
       - **IF Technical/Code:** Focus on logic and functions. Wrap any code/queries in ```markdown code blocks```. Do NOT simulate outputs.
       - **IF Narrative/Text:** Focus on themes, plot points, or key arguments. Use standard bullet points.
    
    2. **Grounding Rules:**
       - **Truth:** Only mention entities (tables, characters, dates) actually present in the text.
       - **No Noise:** Ignore page numbers or artifacts like "Week 3 Lecture 1".
    
    3. **Format:**
       - Use **Headings** to separate ideas.
       - Use **Bullet Points** for readability.
       - Keep descriptions concise.
    ----------------

    Answer:
    """

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=700,
        do_sample=True,  # Essential for natural flow (prevents loops)
        temperature=0.4,  # Stability Sweet Spot (0.1 loops, 0.7 hallucinates)
        top_k=50,  # <--- THE CRITICAL FIX: Restricts vocab to top 50 words to stop hallucinations
        top_p=0.95,
        repetition_penalty=1.1,  # Standard stability penalty
    )

    chat_model = ChatHuggingFace(llm=llm)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = create_stuff_documents_chain(chat_model, prompt)
    return chain


def extract_filename_for_sorting(doc):
    match = re.search(r"--- SOURCE DOCUMENT: (.*?) ---", doc.page_content)
    return match.group(1) if match else "z_unknown"


def user_input(user_question):
    if st.session_state.vector_store is None:
        return "Please process the document first."

    # k=7 gives ~4900 chars context, fitting safely within Zephyr's window
    docs = st.session_state.vector_store.similarity_search(user_question, k=7)
    docs = sorted(docs, key=extract_filename_for_sorting)

    chain = get_conversational_chain()

    response = chain.invoke({"context": docs, "question": user_question})

    return (
        response
        if isinstance(response, str)
        else response.get("output_text", str(response))
    )


def main():
    st.set_page_config("Chat PDF")
    st.header("RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a Question from the PDF Files"):
        if st.session_state.vector_store is None:
            st.error("Please process the document first.")
        else:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", accept_multiple_files=True, type=["pdf"]
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("Vector Store Created Successfully")
            else:
                st.warning("Please upload at least one PDF.")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
