Document Chat App

A web application built with Flask and LangChain to chat with local documents (PDFs, CSVs, Excel files), including support for image-based PDFs via OCR. Powered by Ollama's Llama 3.1 model, it allows users to query document content through a simple interface.

- Features
  * Supported File Types: PDFs (text and image-based), CSVs, Excel (.xlsx, .xls)
  * OCR Support: Extracts text from image-based PDFs using Tesseract
  * Dynamic Updates: Automatically detects and indexes new or changed documents
  * Web Interface: Chat with documents via a browser-based UI
  * Optimized Retrieval: Uses embeddings (`all-mpnet-base-v2`) and a custom prompt for employment-related queries
  * Debugging Tools: Endpoint to inspect loaded documents

- Prerequisites
  * Python: 3.13 (tested, but 3.12+ should work)
  * System Dependencies:
    - On Windows:
      - Poppler: Install via `conda install -c conda-forge poppler` or download from a binary source and add to PATH
      - Tesseract: Download from https://github.com/UB-Mannheim/tesseract/wiki, install, and add to PATH
    - On macOS:
      - Poppler: `brew install poppler`
      - Tesseract: `brew install tesseract`
  * Ollama: Installed and running with the `llama3.1` model pulled (`ollama pull llama3.1`)

- Installation
  1. Clone the Repository:
    git clone https://github.com/yourusername/document-chat-app.git
    cd document-chat-app

2. Set Up a Virtual Environment:
- On Windows:
    python -m venv venv
    venv\Scripts\activate
- On macOS/Linux:
    python3 -m venv venv
    source venv/bin/activate

3. Install Python Dependencies:
    pip install -r requirements.txt
        See `requirements.txt` for the full list of packages.

4. Start Ollama:
    From Terminal Window: ollama serve

Ensure your selected model is available (`ollama list`).

- Folder Structure
document-chat-app/
|-- app.py              Main application script
|-- requirements.txt    Python dependencies
|-- static/            Static files (CSS, JS)
|   `-- style.css      Styling for the web interface
|-- templates/         HTML templates
|   `-- index.html     Web interface HTML
|-- documents/         Place your PDFs, CSVs, Excel files here
`-- vector_store/      Persistent storage for document embeddings (auto-generated)

- Usage
1. Add Documents:
* Place your documents (e.g., `EmploymentContract.pdf`) in the `documents/` folder.
2. Run the App:
    python app.py
    * The app runs on `http://localhost:5001`.

3. Access the Web Interface:
    * Open `http://localhost:5001` in your browser.
    * Type a query (e.g., "What are the terms and conditions of my employment?") and press Enter or click Send.

4. API Usage:
     Query via curl:
    curl -X POST -H "Content-Type: application/json" -d '{"query":"What is in the documents?"}' http://localhost:5001/chat
    * Response includes the answer and source document chunks.

5. Refresh Vector Store:    
    * If you add new documents while the app is running:
        curl -X POST http://localhost:5001/refresh
    * Or restart the app to auto-detect changes.

6. Debug Documents:
    * Check loaded documents:
        curl http://localhost:5001/debug/documents

 
- Configuration
* Chunking: Adjusted to `chunk_size=300, chunk_overlap=50` for precise answers (edit in `app.py`).
* Retrieval: Retrieves top 10 chunks (`k=10`) for better context (edit in `app.py`).
* Prompt: Optimized for employment contracts (customize in `setup_qa_system()`).
* Embedding Model: Uses `all-mpnet-base-v2` (change in `EMBEDDING_MODEL_NAME`).
* Device: Auto-detects CUDA on Windows, falls back to CPU (edit `DEVICE` if needed).

- Performance Optimization
* Smaller chunks and more retrieved documents improve precision.
* Custom prompt focuses on employment-related queries.
* OCR DPI set to 300 for better image-based PDF extraction.
* Logs track retrieved document count per query for evaluation.
To further enhance:
* Adjust `chunk_size` or `k` based on your documents.
* Experiment with MMR retrieval (`search_type="mmr"`) or a larger model (e.g., `llama3.1:70b`).

- Troubleshooting
* No Text Extracted: Ensure documents are in `documents/` and are readable (text or scannable images).
* Port Conflict: Change `port=5001` in `app.run()` if 5001 is in use.
* OCR Fails: Verify Poppler and Tesseract are in PATH (Windows: `where poppler`, `where tesseract`).
* CUDA Issues: Ensure PyTorch with CUDA is installed if using GPU (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).

- Contributing
* Fork the repo, make changes, and submit a pull request.
* Report issues or suggest features via GitHub Issues.

- License
MIT License - feel free to use, modify, and distribute.

- Acknowledgments
* Built with Flask, LangChain, and Ollama.
* Thanks to the open-source community for the underlying tools.       




Windows Setup Instructions

Install Python 3.13:
    Download from python.org and ensure it’s added to PATH.

Install System Dependencies:
    Poppler: Download from a source like this GitHub repo, extract, and add bin/ to your system PATH.
    Tesseract: Download from UB-Mannheim, install, and add to PATH.
    Verify: Open CMD and run tesseract --version and pdftoppm -v.

Install Ollama:
Download from ollama.ai, install, and run:
    ollama pull llama3.1
    ollama serve

Set Up and Run:
    Clone the repo, create the virtual environment, install dependencies, and run as per the README.
Optional CUDA Support:
    If you have an NVIDIA GPU, install PyTorch with CUDA:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
The code will auto-detect and use CUDA.





