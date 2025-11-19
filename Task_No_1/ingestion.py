import os
from unittest import loader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])


# --- . DEFINE PDFS AND DB DIRECTORY ---

file_paths =[
    "./Data/2506.02153v2.pdf",
    "./Data/reasoning_models_paper.pdf"
]
CHROMA_DB_DIRECTORY = ".\chroma_db2"

def main():

    # ---  LOAD DOCUMENTS ---
    
    pages = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path)
        for doc in loader.lazy_load():
            pages.append(doc)

    print(f"Loaded {len(pages)} document pages.")

    # --- SPLIT TEXT INTO CHUNKS ---
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Split documents into {len(chunks)} chunks.")

    # --- CREATE OR LOAD VECTOR STORE ---  
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_DB_DIRECTORY
    )

    retriever = db.as_retriever(search_kwargs={"k": 2}) 
    
if __name__ == "__main__":
    
    if not os.getenv("GOOGLE_API_KEY"):
       
        print("Warning: GOOGLE_API_KEY environment variable not set, but we are using HuggingFace.")
        

    main()
