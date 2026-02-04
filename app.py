import streamlit as st
import pymupdf4llm
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
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
    model = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
    )
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
    # 1. Display Chat History FIRST
    # (This ensures old messages are shown before we check for new input)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Chat Input (Pins to Bottom)
    # The ':= ' operator assigns the input to 'prompt' AND checks if it's not empty
    if prompt := st.chat_input("Ask a Question from the PDF Files"):
        # Check if PDF is processed
        if st.session_state.vector_store is None:
            st.error("Please process the document first.")
        else:
            # A. Display User Message Immediately
            st.chat_message("user").markdown(prompt)
            # Add to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # B. Get AI Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.markdown(response)
            # Add AI response to history
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar - PDF Upload & Processing
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
