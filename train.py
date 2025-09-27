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

# Step 1: Define the prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}\n\nContext:\n{context}")
])

# Step 2: Create the LCEL chain
rag_chain = (
    RunnableMap({
        "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
        "question": RunnablePassthrough()
    })
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Step 3: Run the chain
query = "Who is Dr. Judea Pearl and what did he warn ML people about in which book?"
response = rag_chain.invoke(query)

# Step 4: Output
print(response)