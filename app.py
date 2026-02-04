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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    prompt_template = prompt_template = """
    You are an expert technical assistant. Use the provided context to answer the question.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    ----------------
    UNIVERSAL PROTOCOL (FOLLOW STRICTLY):
    
    1. **Identify Document Type & Adapt Strategy:**
       - **IF Technical/Scientific (SQL, Code, Medical):** Explain the *purpose* and *logic* of the code/queries. (e.g., "This query calculates average salary...").
       - **IF Narrative (Literature, History):** Focus on the *plot, timeline, themes, and key characters*.
       - **IF Legal/Financial:** Focus on *obligations, dates, and clauses*.

    2. **CRITICAL: NO SIMULATION OR HALLUCINATION:**
       - **NEVER** invent "Output" or "Result" tables. If the text shows a SQL query, explain what it *does*, do NOT pretend to run it.
         - BAD: "Output: | dept_name | 12 |"
         - GOOD: "The query aggregates credits by department."
       - Do not mention specific data values (like "$5000" or "12 credits") unless they are explicitly in the text as examples.

    3. **Noise Filtration:**
       - **Ignore Page Artifacts:** Numbers like "10", "11", "Week 3" at the ends of sections are Page Numbers. Do NOT treat them as new topics.
       - **Ignore Repetitive Lists:** Summarize lists (e.g., "A list of course codes") rather than reproducing them.

    4. **Handling Tables:**
       - **Interpret, Don't Copy:** Never copy-paste table rows. Explain the *relationship* the table demonstrates.

    5. **Logical Synthesis:**
       - Connect ideas using transition words. Do not output disjointed facts.
       - Treat distinct topics (e.g., "Week 1", "Week 2") as separate sections.

    6. **Formatting Rules:**
       - Use **Headings** for major sections.
       - Use **Bullet Points** for lists.
       - Use **Bold** for key terms.

    7. **Length Constraint:**
       - Keep the summary concise (under 600 words) unless requested otherwise.
    ----------------
    
    Answer:
    """

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=1.3,
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

    docs = st.session_state.vector_store.similarity_search(user_question, k=4)
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
