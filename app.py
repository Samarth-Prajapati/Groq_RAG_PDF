# Importing Dependencies
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
import tempfile

# Load .env
load_dotenv()

# API Key
groq_api_key = os.getenv('GROQ_API_KEY')

# LLM Model
llm = ChatGroq(api_key = groq_api_key, model = 'llama-3.1-8b-instant')

# Chat Prompt Template
prompt = ChatPromptTemplate.from_template(
    '''
    You are an AI assistant.
    Answer the question strictly using the given context.
    If the answer is not present in the context, say:
    "The answer is not available in the provided document."
    
    <context> {context} </context>
    
    Question : {question}
    '''
)

# Output Parser
parser = StrOutputParser()

def save_uploaded_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        return tmp.name

# Create Embeddings and Store in Vector DB
def create_vector_embeddings(file_path):
    if 'vectors' not in st.session_state:
        pdf_path = save_uploaded_pdf(file_path)

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFLoader(pdf_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

# Streamlit Interface
st.title('Q&A RAG APP ( PDF )')

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    create_vector_embeddings(uploaded_file)
    st.success('Vector DB created successfully, ready to answer all of your questions.')

    query = st.text_input('Enter your query')
    if query:
        retriever = st.session_state.vectors.as_retriever()

        chain = ({'context' : retriever, 'question' : RunnablePassthrough()}
                 | prompt
                 | llm
                 | parser)

        start_time = time.process_time()
        response = chain.invoke(query)
        print(f'Response time : {time.process_time() - start_time}')

        st.write(response)
