# Permission-Aware Multi-Tenant RAG System

A clean, minimal dashboard for a permission-aware multi-tenant RAG system built with Flask, ChromaDB, and Ollama.

## Project Structure

```
permission-aware-rag/
├── app.py                    # Flask entry point and API routes
├── users.json                # Existing development user registry
├── requirements.txt          # Existing and required dependencies
├── templates/
│   └── index.html            # Main dashboard
├── static/
│   ├── css/
│   │   └── style.css         # Dashboard styling
│   └── js/
│       └── app.js            # Chat interactions and API calls
├── src/
│   ├── retrieval.py          # Existing RAG retrieval and security
│   ├── generation.py         # Existing Ollama generation
│   ├── relevance.py          # Existing relevance and access checks
│   ├── chroma_setup.py       # Database setup
│   ├── embeddings.py         # Embedding generation
│   ├── chunking.py           # Text chunking
│   └── ingest.py             # Document ingestion
├── data/
│   ├── chunks/               # Chunk metadata (chunks.json)
│   ├── embeddings/           # Stored vectors (vectors.npy, metadata.json)
│   ├── processed/            # Processed document texts
│   ├── public/               # Source documents (PDFs/DOCXs)
│   └── chroma_db/            # ChromaDB vector database
├── tests/
│   ├── test.py               # Existing citation validation tests
│   ├── test_relevance.py     # Existing relevance filter tests
│   ├── test_retrieval_security.py  # Existing security tests
│   └── test_flask_api.py     # New Flask API tests
└── README.md                 # This file
```

## Setup Instructions

### 1. Activate the existing virtual environment

```bash
source .venv/bin/activate
```

### 2. Install missing dependencies

```bash
pip install flask
```

(Other dependencies like sentence-transformers, numpy, pypdf, and python-docx should already be installed from the existing environment.)

### 3. Start Ollama and verify the configured model is available

This project uses Ollama with the `gemma3:4b` model. Ensure Ollama is running and the model is available:

```bash
ollama serve
ollama pull gemma3:4b
```

### 4. Configure the development user

Set the `RAG_USER_ID` environment variable to select your authorized user identity. The available users are defined in `users.json`:

```bash
# For Tenant A internal access
export RAG_USER_ID=employee_a

# For Tenant B internal access
export RAG_USER_ID=employee_b

# For public access
export RAG_USER_ID=public_analyst
```

### 5. Start Flask using the appropriate command

Start the Flask development server:

```bash
python app.py
```

The server will be available at `http://localhost:5001` (or the `PORT` environment variable if set).

### 6. Open the local dashboard URL in a browser

Navigate to `http://localhost:5001` to open the SecureRAG dashboard.

### 7. Test an authorized query

1. Enter a question related to your authorized tenant's documents
2. Example (Tenant A): "What is Tenant A's MFA policy?"
3. The system will retrieve authorized chunks, generate an answer, and display source citations

### 8. Test a cross-tenant request

1. With `RAG_USER_ID=employee_a`, ask a question targeting Tenant B
2. Example: "What are Tenant B's security requirements?"
3. The system will respond: "This request is outside your authorized information boundary."

### 9. Common errors and their fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `400: Questions cannot be empty` | Empty question submitted | Enter a non-empty question |
| `403: Development identity not configured` | `RAG_USER_ID` not set | Set `RAG_USER_ID` environment variable |
| `503: The local AI service could not be reached` | Ollama not running | Start Ollama and ensure `gemma3:4b` is available |
| `access_denied` | Cross-tenant question | Ask about your own tenant or public documents |
| `insufficient_evidence` | Question has no relevant authorized documents | Rephrase to ask about public reports or your organization's policies |

## Security Notes

- The `RAG_USER_ID` environment variable is **not production authentication**. It is a development convenience for local prototyping.
- Identity is always resolved on the server via `users.json`. The browser should never supply `tenant_id` or `user_id`.
- Cross-tenant requests are blocked server-side. The frontend cannot bypass these checks.
- Unauthorized document metadata is never returned to the browser.

## Existing Tests

Run the full test suite:

```bash
PYTHONPATH=. pytest tests/ -v
```

All 35 unit, integration, and security tests should pass. Tests are organized under `tests/`.

---

## Spatial 3D UI Workspace (Three.js + React Three Fiber)

SecureRAG features an enterprise-grade spatial 3D AI workspace built with **React**, **Three.js**, **React Three Fiber (`@react-three/fiber`)**, and **`@react-three/drei`**.

### Frontend Features
- **3D Spatial Atmosphere**: Three.js Canvas with subtle ambient navy illumination, perspective depth grid, floating knowledge nodes, and gentle cursor parallax.
- **Accessible Glass Panels**: High-contrast, selectable, DOM glassmorphic overlays with backdrop blur.
- **Workspace Identity**: Realistic presentation mapping (Priya Sharma / Northstar Financial) connected to trusted server-side identity.
- **Interactive Citations**: Inline `[S1]` badge clicks automatically highlight corresponding source cards.
- **Distinct State Cards**: Customized cards for Insufficient Evidence, Tenant Isolation Access Restrictions, and Ollama service errors.
- **Non-WebGL Fallback**: Automatic detection and graceful fallback if WebGL is unavailable.

### Running Both Servers

#### Terminal 1 — Start the Flask Backend (Port 5001)
```bash
source .venv/bin/activate
export RAG_USER_ID=employee_a
python3 app.py
```

#### Terminal 2 — Start the Spatial 3D Frontend (Port 5173)
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/` in your browser. All `/api` requests are automatically proxied to Flask on port 5001.