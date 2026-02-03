import streamlit as st
import pymupdf4llm
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.getbuffer())
            tmp_path = tmp_file.name
        try:
            text += pymupdf4llm.to_markdown(tmp_path)
        finally:
            os.remove(tmp_path)
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say "answer is not available in the context", do not provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = create_stuff_documents_chain(model, prompt)
    return chain


def user_input(user_question):
    if st.session_state.vector_store is None:
        return "Please process the document first."

    docs = st.session_state.vector_store.similarity_search(user_question)
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

    def process_input():
        user_question = st.session_state["user_input_box"]
        if user_question:
            if st.session_state.vector_store is not None:
                # Append user question to history
                st.session_state.messages.append(
                    {"role": "user", "content": user_question}
                )
                response = user_input(user_question)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.session_state["user_input_box"] = ""

            else:
                st.error("Please process the document first.")

    st.text_input(
        "Ask a Question from the PDF Files",
        key="user_input_box",
        on_change=process_input,
    )

    # Sync chat history with the UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
