# Vectorless-RAG

A vectorless Retrieval-Augmented Generation (RAG) platform using autonomous LLM agents to perform one-shot compressed tree search reasoning.

## Problem Statement and Solution Overview

**Problem:** Traditional RAG systems rely heavily on vector databases (like Pinecone or FAISS) to calculate cosine similarity between document chunks and user queries. This approach often fails to understand the structural context of the document (e.g., matching a keyword in a sub-bullet point without knowing its parent heading) and can result in fragmented or incorrect context retrieval.

**Solution:** This project completely abandons vector embeddings. Instead, it extracts the hierarchical structure of a document (e.g., Chapters -> Sections -> Body) using a custom PDF parser. An autonomous LLM agent then evaluates a compressed version of this "Document Tree" in a single prompt, instantly identifying the exact structural nodes that contain the answer. This ensures highly accurate, structurally-aware context retrieval with minimal API calls.

## Tech Stack and Decisions

* **Python 3.x:** Core programming language.
* **Streamlit:** Chosen for rapid, interactive frontend prototyping to visualize real-time agent reasoning paths without heavy frontend boilerplate.
* **Groq API (Llama-3-70b-versatile)** 
* **PyMuPDF (fitz):** Selected for PDF processing because it provides block and span-level details, enabling dynamic detection of font-size hierarchies.
* **Tenacity:** Used to implement robust exponential backoff and retry logic, ensuring the system elegantly handles rate-limits during high-speed tree generation.

## Features

- [x] One-Shot Compressed Tree Search Routing
- [x] Custom PDF Parsing with Dynamic Font-Size Hierarchy Detection
- [x] Zero-latency Regex Heuristics for Cost/Speed Optimization
- [x] Exponential Backoff for Rate Limiting Resilience
- [x] O(1) API Calls per Query (Massive Speed & Cost Efficiency)
- [x] Real-time Interactive Traversal UI (Streamlit)
- [x] LLM Response Cleaning for JSON Parsing Resilience
- [ ] Multi-Document Knowledge Base Support
- [ ] Exportable Agent Reasoning Logs

## Getting Started

### Prerequisites

* Python 3.9+
* Pip (Python package manager)
* A Groq API Key

### Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pageIndex_Rag
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables

Copy the example environment file and add your credentials:
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_actual_api_key_here
```

### Run Scripts

**Development / UI Mode:**
Start the interactive Streamlit application to upload resumes and chat with the agent:
```bash
streamlit run app.py
```

**Production / CLI Mode (Headless Indexing):**
Generate a JSON tree index manually without starting the UI:
```bash
python generate_index.py "path/to/your/document.pdf" -o "tree_index.json"
```

## Deployment

To deploy this application:
1. Push the code to a GitHub repository.
2. Connect the repository to [Streamlit Community Cloud](https://streamlit.io/cloud) or [Render](https://render.com).
3. Set the `GROQ_API_KEY` in your hosting provider's Secrets/Environment Variables configuration panel.
4. Deploy using `app.py` as the entry point.

## Project Structure

```text
├── .env.example                # Example environment variables
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── app.py                      # Main Streamlit application and UI logic
├── agents.py                   # NavigationAgent logic and LLM routing algorithms
├── indexer.py                  # Document tree construction and LLM summarization
├── pdf_tools.py                # PyMuPDF structured extraction and hierarchy detection
└── generate_index.py           # CLI script for headless document indexing
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.
