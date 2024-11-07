import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up Streamlit
st.title("Simple RAG Application")
def hide_streamlit_style():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

hide_streamlit_style()

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# File upload for PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Process PDF and create vector store
if uploaded_file:
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize embeddings and document loader
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFLoader("uploaded_document.pdf")
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])

    # Create FAISS vector store
    vectors = FAISS.from_documents(final_documents, embeddings)
    retriever = vectors.as_retriever()

    # Define retrieval prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Question: {input}
        """
    )

    # Set up document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get user query
    user_query = st.text_input("Enter your question here:")

    if user_query:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_query})
        end_time = time.process_time() - start
        st.write(f"Response time: {end_time:.2f} seconds")
        st.subheader("AI Response:")
        st.write(response['answer'])

        # Display retrieved documents with similarity search
        with st.expander("Document Similarity Search Results"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Result {i + 1}:**")
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.warning("Please upload a PDF file to start.")
