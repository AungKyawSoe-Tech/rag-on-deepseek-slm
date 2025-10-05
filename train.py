# Import

import os
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama
#from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser


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

# Main()  

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