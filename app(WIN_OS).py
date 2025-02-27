# Import necessary libraries
from flask import Flask, render_template, request, jsonify  # Flask for web app functionality
import os  # For file system operations
import shutil  # For directory manipulation (e.g., deleting vector store)
import json  # For reading/writing JSON files (hashes and model info)
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader  # Loaders for different file types
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Updated: Generates embeddings for text
from langchain_chroma import Chroma  # Updated: Vector store for similarity search
from langchain_ollama import OllamaLLM  # Updated: Interface to Ollama language model
from langchain.chains import RetrievalQA  # QA chain for question answering
from langchain.prompts import PromptTemplate  # Custom prompt for better answers
from langchain_core.documents import Document  # Document object for LangChain
from pathlib import Path  # Modern file path handling (cross-platform)
import logging  # For logging messages
import hashlib  # For generating file hashes
from pdf2image import convert_from_path  # Converts PDF pages to images for OCR
import pytesseract  # OCR library to extract text from images
import platform  # To detect operating system

# Set up logging to display info and error messages
logging.basicConfig(level=logging.INFO)  # Log INFO level and above
logger = logging.getLogger(__name__)  # Logger instance for this module

# Initialize Flask app
app = Flask(__name__)

# Define directory paths using Path for cross-platform compatibility
DOCS_DIR = Path("documents")  # Where input documents (PDFs, CSVs, Excel) are stored
VECTOR_DIR = Path("vector_store")  # Where the vector store and metadata are saved
DOCS_DIR.mkdir(exist_ok=True)  # Create documents directory if it doesn't exist
VECTOR_DIR.mkdir(exist_ok=True)  # Create vector store directory if it doesn't exist

# Files for tracking document changes and embedding model
DOC_HASH_FILE = VECTOR_DIR / "doc_hashes.json"  # Stores hashes of documents to detect changes
MODEL_INFO_FILE = VECTOR_DIR / "model_info.json"  # Stores the embedding model name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Embedding model for text (768 dimensions)

# Detect operating system for device selection
OS = platform.system()
DEVICE = "cuda" if OS == "Windows" and torch.cuda.is_available() else "cpu"  # Use CUDA if available on Windows, else CPU

def get_file_hash(file_path):
    """Generate an MD5 hash of a file to check if it has changed."""
    hasher = hashlib.md5()  # Initialize MD5 hash object
    with open(file_path, "rb") as f:  # Open file in binary mode
        hasher.update(f.read())  # Update hash with file contents
    return hasher.hexdigest()  # Return hexadecimal hash

def load_pdf_with_ocr(file_path):
    """Extract text from image-based PDFs using OCR."""
    try:
        # Convert PDF pages to images (requires poppler installed)
        images = convert_from_path(file_path, dpi=300)  # Increased DPI for better OCR
        documents = []  # List to store extracted Document objects
        
        # Process each page/image
        for i, image in enumerate(images):
            # Use Tesseract to extract text from the image
            text = pytesseract.image_to_string(image)
            if text.strip():  # Check if any text was extracted
                # Create a Document object with the text and metadata
                doc = Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": i + 1}
                )
                documents.append(doc)
            else:
                logger.warning(f"No text extracted from page {i+1} of {file_path.name}")
        
        logger.info(f"Loaded {len(documents)} pages from {file_path.name} via OCR")
        return documents
    except Exception as e:
        logger.error(f"Failed to process {file_path.name} with OCR: {str(e)}")
        return []  # Return empty list on failure

def load_documents(existing_hashes=None):
    """Load documents from the 'documents/' folder, handling new/changed files."""
    documents = []  # List to store all loaded documents
    new_hashes = {}  # Dictionary to store hashes of processed files
    
    # Check if the documents directory exists and has files
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        logger.warning("No documents found in 'documents/' folder.")
        return documents, new_hashes
    
    # Iterate over all files in the documents directory
    for file_path in DOCS_DIR.glob("*"):
        try:
            # Generate hash to detect changes
            file_hash = get_file_hash(file_path)
            new_hashes[str(file_path)] = file_hash
            
            # Skip if file hasn’t changed (hash matches existing)
            if existing_hashes and str(file_path) in existing_hashes and existing_hashes[str(file_path)] == file_hash:
                continue
            
            # Handle different file types
            if file_path.suffix == ".pdf":
                # First try standard PDF text extraction
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                # If no text is extracted, assume it’s image-based and use OCR
                if not any(doc.page_content.strip() for doc in docs):
                    logger.info(f"No text extracted from {file_path.name} with PyPDFLoader; trying OCR.")
                    docs = load_pdf_with_ocr(file_path)
                else:
                    logger.info(f"Loaded {len(docs)} pages from {file_path.name} with PyPDFLoader")
                documents.extend(docs)
            elif file_path.suffix == ".csv":
                loader = CSVLoader(str(file_path))
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} entries from {file_path.name}")
                documents.extend(docs)
            elif file_path.suffix in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(str(file_path))
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} entries from {file_path.name}")
                documents.extend(docs)
            else:
                logger.warning(f"Unsupported file type: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {str(e)}")
    return documents, new_hashes  # Return loaded docs and their hashes

def initialize_or_update_vector_store():
    """Set up or update the vector store with document embeddings."""
    # Create embeddings generator (converts text to numerical vectors)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE}  # Use detected device (cuda/cpu)
    )
    
    # Load existing document hashes and embedding model info
    existing_hashes = {}
    stored_model = None
    if DOC_HASH_FILE.exists():
        with open(DOC_HASH_FILE, "r") as f:
            existing_hashes = json.load(f)
    if MODEL_INFO_FILE.exists():
        with open(MODEL_INFO_FILE, "r") as f:
            stored_model = json.load(f).get("embedding_model")
    
    # Load new or changed documents
    docs, new_hashes = load_documents(existing_hashes)
    
    # If no documents are loaded and no prior data exists, fail
    if not docs and not existing_hashes:
        logger.error("No documents loaded and no existing vector store.")
        raise ValueError("No documents available to process.")
    
    # If the embedding model changed, clear the old vector store
    if stored_model != EMBEDDING_MODEL_NAME:
        if VECTOR_DIR.exists():
            shutil.rmtree(VECTOR_DIR)
            logger.info("Cleared vector store due to embedding model change.")
        VECTOR_DIR.mkdir()
    
    # Decide whether to load existing store or update it
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()) and not docs and stored_model == EMBEDDING_MODEL_NAME:
        vector_store = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)
        logger.info("Loaded existing vector store with no new documents.")
    else:
        # Split documents into smaller chunks for better retrieval precision
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs) if docs else []
        logger.info(f"Split new documents into {len(chunks)} chunks")
        
        # Log a sample chunk for debugging
        if chunks:
            logger.info(f"Sample chunk content: {chunks[0].page_content[:200]}")
        
        # Update or initialize the vector store
        if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()) and stored_model == EMBEDDING_MODEL_NAME:
            vector_store = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)
            if chunks:
                vector_store.add_documents(chunks)  # Add new chunks to existing store
                logger.info("Updated existing vector store with new documents.")
        else:
            # If no chunks and no prior data, fail
            if not chunks and not existing_hashes:
                logger.error("No chunks to initialize vector store.")
                raise ValueError("No document chunks available to process.")
            # Initialize with all documents if starting fresh
            vector_store = Chroma.from_documents(chunks or load_documents()[0], embeddings, persist_directory=str(VECTOR_DIR))
            logger.info("Initialized new vector store.")
        
        vector_store.persist()  # Save to disk
    
    # Save updated hashes and model info
    with open(DOC_HASH_FILE, "w") as f:
        json.dump(new_hashes, f)
    with open(MODEL_INFO_FILE, "w") as f:
        json.dump({"embedding_model": EMBEDDING_MODEL_NAME}, f)
    
    return vector_store

# Global variables for vector store and QA chain
vector_store = None
qa_chain = None

def setup_qa_system():
    """Initialize the vector store and QA system."""
    global vector_store, qa_chain
    vector_store = initialize_or_update_vector_store()
    
    # Test the vector store with a sample query
    test_query = "What is in the documents?"
    retrieved_docs = vector_store.similarity_search(test_query, k=10)  # Increased to 10 for more context
    logger.info(f"Retrieved {len(retrieved_docs)} documents for test query: {test_query}")
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Retrieved doc {i+1}: {doc.page_content[:200]}")
    
    # Initialize the language model (Ollama with Llama 3.1)
    llm = OllamaLLM(model="llama3.1")
    
    # Define a custom prompt tailored for employment-related queries
    prompt_template = """You are an expert in employment contracts. Use the following context from employment-related documents to answer the question accurately and concisely. If the answer isn’t in the context, say "I couldn’t find the answer in the documents." Do not invent details.

Context: {context}

Question: {question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Set up the QA chain with retrieval
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple method: stuffs all context into prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),  # Increased to 10 for more context
        return_source_documents=True,  # Include source docs in response
        chain_type_kwargs={"prompt": prompt
