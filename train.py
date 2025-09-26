# Import

import os

from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings

import streamlit as st  

# color palette
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#4561e9"

# Functions
def load_text_documents(directory, documents):
   
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                documents.append(file.read())
  

def load_pdf_documents(directory, documents):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            print(directory + '/' + filename)
            with open(os.path.join(directory, filename), 'r') as file:
                documents.append(PDFPlumberLoader(directory + '/' + filename).load())


# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)



# Main()  

# Streamlit app title
st.title("Build a RAG System with DeepSeek R1 & Ollama")

# Loading documents

loaded_documents = []

load_text_documents('../Documents', loaded_documents)
load_pdf_documents('../Documents', loaded_documents)

doc_counter = 0

for doc in loaded_documents :
    # Split text into semantic chunks 
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())  
    documents = text_splitter.split_documents(doc)  

    # Generate embeddings  
    embeddings = HuggingFaceEmbeddings()  
    vector_store = FAISS.from_documents(documents, embeddings)  

    print("done processing doc #" + str(doc_counter))
    doc_counter = doc_counter + 1

print("RAG system has processed loaded documents!")

# Connect retriever  
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks  

llm = Ollama(model="deepseek-r1:1.5b")  

print("Craft the prompt template below:")

# Craft the prompt template  
prompt = """  
1. Use ONLY the context below.  
2. If unsure, say "I don’t know".  
3. Keep answers under 4 sentences.  

Context: {context}  

Question: {question}  

Answer:  
"""  
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)  

# Chain 1: Generate answers  
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)  

# Chain 2: Combine document chunks  
document_prompt = PromptTemplate(  
    template="Context:\ncontent:{page_content}\nsource:{source}",  
    input_variables=["page_content", "source"]  
)  

# Final RAG pipeline  
qa = RetrievalQA(  
    combine_documents_chain=StuffDocumentsChain(  
        llm_chain=llm_chain,  
        document_prompt=document_prompt  
    ),  
    retriever=retriever  
)  

# Streamlit UI  
# May be next step is to make this model run as a server and send queries wih curl

print("Asking RAG a question:")

user_input = st.text_input("Ask your RAG a question:")  

if user_input:  
    with st.spinner("Thinking..."):  
        response = qa(user_input)["result"]  
        st.write(response) 