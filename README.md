# 📄 Searchable PDF Q&A Application

This project is a **Retrieval-Augmented Generation (RAG)** application that allows you to *chat* with your PDF documents. It uses a **Streamlit web interface** to ask questions and get answers sourced directly from the provided texts.

The backend is built with **LangChain**, using **HuggingFace embeddings** to understand the text, **ChromaDB** as a vector database to store the knowledge, and **Google's Gemini (Gemini 2.5 Flash)** model to generate answers.

---


## 🚀 How It Works

The application operates in two main stages:

### 1. Ingestion (`ingestion.py`)

* Reads your PDF documents from the `./Data` folder.
* Uses `PyMuPDFLoader` to load pages.
* Splits text into smaller, manageable chunks (`RecursiveCharacterTextSplitter`).
* Converts each chunk into a numerical vector (embedding) using HuggingFace model `all-MiniLM-L6-v2`.
* Stores these vectors in a **persistent ChromaDB** inside the `chroma_db2` directory.
* This process only needs to be done once (or when your documents change).

### 2. Application (`app.py`)

* Runs a **Streamlit web server**.
* Loads the pre-built Chroma database and initializes the **Google Gemini LLM**.
* When you ask a question:

  * It converts your question into an embedding.
  * Queries the Chroma database for the most relevant text chunks (retrieval step).
  * Passes your question and retrieved context to Gemini, which generates an answer based on the context.

---

## ⚙️ Setup & Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/codeagrawal07/JD-Fusion-Tasks-GENAI-RAG.git
cd JD-Fusion-Tasks-GENAI-RAG
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

This project requires a Google API key for Gemini.

Create a new `.env` file in the project root and add your API key:

```env
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

---

## 🏃‍♂️ How to Run the Application

You must run the ingestion script first to build the database, then start the app.

### Step 1: Ingest Your Documents

Add the PDF files you want to query into the `Data/` folder, then run:

```bash
python ingestion.py
```

You’ll see logs showing the document loading and embedding process. A folder named `chroma_db2` will be created.

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

Streamlit will open in your browser (usually [http://localhost:8501](http://localhost:8501)). You can now start asking questions!

---

## 🗂️ Project File Structure

```
.
├── Data/                 # Folder to store source PDF documents
│   ├── 2506.02153v2.pdf
│   └── reasoning_models_paper.pdf
├── .env                  # (You must create this) Stores API keys
├── .gitignore            # Git ignore file
├── app.py                # The main Streamlit Q&A application
├── ingestion.py          # Script to load and embed documents into ChromaDB
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 💡 Key Technologies Used

* **LangChain** – For RAG pipeline and chaining components.
* **HuggingFace Transformers** – For text embeddings (`all-MiniLM-L6-v2`).
* **ChromaDB** – As the vector database.
* **Streamlit** – For web interface.
* **Google Gemini 2.5 Flash** – As the LLM for contextual question answering.
* **PyMuPDF** – For extracting text from PDFs.

---

## 📈 Future Improvements

* Add multi-file support for cross-document querying.
* Implement caching to reduce repeated API calls.
* Add support for other LLMs (OpenAI, LLaMA, etc.).
* Include file upload functionality in the Streamlit interface.

---

### 💎 Author

**Abhishek Agrawal**
GitHub: [codeagrawal07](https://github.com/codeagrawal07)
