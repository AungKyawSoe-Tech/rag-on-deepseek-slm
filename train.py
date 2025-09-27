# Import

import os
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
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

load_text_documents('Documents', loaded_documents)
load_pdf_documents('Documents', loaded_documents)

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
QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(prompt)  

# Step 1: Define the prompt for combining documents

# This is the main prompt that wraps all documents
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following documents:\n\n{context}")
])

document_prompt = ChatPromptTemplate(
    template="Context:\ncontent:{page_content}\nsource:{source}",
    input_variables=["page_content", "source"],
    messages=[
          ("system", "You are a helpful AI assistant."),
          ("human", "{question}"),
      ]
)

# Step 2: Create the document stuffing chain
stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt, document_prompt=document_prompt)

# Step 3: Compose the RetrievalQA chain using LCEL
retrieval_chain = create_retrieval_chain(retriever, stuff_chain)

# Streamlit UI  
# May be next step is to make this model run as a server and send queries wih curl

print("Asking RAG a question:")

user_input = st.text_input("Ask your RAG a question:")  

if user_input:  
    with st.spinner("Thinking..."):  
        response = qa(user_input)["result"]  
        st.write(response) 