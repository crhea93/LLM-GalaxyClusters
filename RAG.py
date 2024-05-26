import os
from dotenv import load_dotenv
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import chromadb
import time

load_dotenv('.env')

def load_chunk_persist_pdf() -> Chroma:
    print('Checking for existing vector database...')
    persist_directory = "./"
    db_path = os.path.join(persist_directory, "chroma.sqlite3")  # Adjust based on your actual DB file name
    print(db_path)
    if os.path.exists(db_path):
        print('Loading existing vector database...')
        vectordb = Chroma(
            embedding_function=HuggingFaceEmbeddings(),
            persist_directory=persist_directory
        )
        return vectordb

    print('No existing vector database found. Creating new vector database...')
    pdf_folder_path = "./"
    documents = []

    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    print('Splitting text...')
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    print('Creating client...')
    client = chromadb.Client()

    if client.list_collections():
        astronomy_collection = client.create_collection("astronomy_collection")
    else:
        print("Collection already exists")

    print('Creating vector database...')
    
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=HuggingFaceEmbeddings(),
        persist_directory=persist_directory
    )
           

    vectordb.persist()
    return vectordb

def create_agent_chain():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    promptTemplate = PromptTemplate.from_template(
                """ You are a helpful assistant that can answer questions based on the given documents and rules.
                 
                You should use the tools below to answer the question posed of you:

                    Rules:
                    A)Only use the factual information from the available PDFs to answer the question.
                    B)Please include references for each sentence in the following format. 
                        Example: Galaxy clusters contain millions of stars[1] [1] Reference Information
                        For the Reference Information please include the author list and the year of publication
                    
                    Question: {task}
                """
        )
    prompt = promptTemplate.format(
                task=query
            )

    answer = chain.run(input_documents=matching_docs, question=prompt)
    return answer


# Streamlit UI
# ===============
st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("Galaxy Clusters")

user_input = st.text_input("Please ask your question about galaxy clusters here")
if user_input:
    with st.spinner("⛏️ Digging through the literature "):
        st.write(get_llm_response(user_input))
