import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# 1. Configuration: Accessing Cloud Secrets [6, 9]
api_key = st.secrets
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)

st.title("📄 IITM BS RAG Chatbot")

# 2. Initialize Memory (Session State)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages =

# 3. Dynamic PDF Upload logic
with st.sidebar:
    st.header("Document Processor")
    pdf_file = st.file_uploader("Upload a PDF to start", type="pdf")
    
    if pdf_file and st.button("Process Document"):
        with st.spinner("Analyzing document..."):

            reader = PdfReader(pdf_file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            

            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            chunks = splitter.split_text(text)
            

            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            st.session_state.vector_store = vector_store
            st.success("Document processed!")

# 4. Chat Interface logic
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the PDF"):
    if not st.session_state.vector_store:
        st.warning("Please upload a PDF in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Retrieval
        docs = st.session_state.vector_store.similarity_search(prompt)
        chain = load_qa_chain(model, chain_type="stuff")
        response = chain.run(input_documents=docs, question=prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})