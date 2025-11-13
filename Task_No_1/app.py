import streamlit as st
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
CHROMA_DB_DIRECTORY = ".\\Task_No_2 \\chroma_db2"

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY environment variable not set. Please set it and restart.")
    st.stop()

# --- 2. LOAD MODELS & DATABASE ---

@st.cache_resource
def load_components():
    # Load the embedding model
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    # Load the persisted vector database
    db = Chroma(
        persist_directory=CHROMA_DB_DIRECTORY, 
        embedding_function=embeddings
    )
    
    # Load the LLM for generating answers
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # Create a retriever
    retriever = db.as_retriever(search_kwargs={"k": 3}) 

    return retriever, llm

# Define the prompt template
PROMPT_TEMPLATE = """
You are a helpful and accurate question-answering assistant.
Task: Answer the user's question precisely and concisely using only the information from the "Context" provided below.
Guidelines:
Your answer must be based solely on the text in the "Context." Do not use any outside knowledge.
Write your answer in a simple and direct tone.
If the context includes relevant examples to support the answer, you may use them.
Crucially: If the answer to the question cannot be found in the "Context," you must respond with:
"The provided context does not contain the information needed to answer this question."

Context: {context}

Question: {question}
"""

try:
    retriever, llm = load_components()
except Exception as e:
    st.error(f"Error loading components: {e}")
    st.stop()

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, 
    input_variables=["context", "question"]
)


# --- 3. BUILD THE RAG CHAIN (LANGCHAIN EXPRESSION LANGUAGE) ---
def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
main_chain = parallel_chain | prompt | llm | StrOutputParser()

# --- 4. STREAMLIT UI SETUP ---
st.title("📄 Searchable PDF Q&A")
st.markdown("Ask a question about the two provided technical papers.")

# Text box for user question
question = st.text_input(
    "Enter your question:", 
    placeholder="What is the main topic of the reasoning models paper?"
)

if question:
    with st.spinner("Searching and generating answer..."):
        # --- 5. EXECUTE QUERY & DISPLAY RESULTS ---
        start_time = time.time() # Start timer
        
        # Get the answer
        answer = main_chain.invoke(question)
        
        end_time = time.time() # End timer
        query_time = end_time - start_time
        
        # Show the final answer
        st.subheader("Answer:")
        st.markdown(answer)
        
        # Show query time [cite: 8]
        st.info(f"Query response time: {query_time:.2f} seconds")
        

