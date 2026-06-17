# LangChain RAG Explorer

This is a local exploratory RAG application using the stack you selected:

```txt
langchain
langchain_community
fastembed
streamlit
streamlit_chat
tinydb
pypdf
qdrant-client
python-dotenv
watchdog
pandas
httpx
```

Qdrant is the vector database. FastEmbed creates embeddings locally. Your already-installed local Ollama is used for answer generation.

## Architecture

```text
Streamlit
  -> UI + app flow

LangChain
  -> loaders, splitters, retriever flow

pypdf
  -> PDF extraction through LangChain PDF loader

FastEmbed
  -> local embeddings

Qdrant
  -> vector storage and similarity search

TinyDB
  -> local document metadata, sessions, chat messages

Ollama
  -> local offline LLM generation

httpx
  -> local HTTP call to Ollama
```

## Flow

```text
Upload PDF/TXT/MD
  -> LangChain loader / pypdf extracts text
  -> LangChain splits into chunks
  -> FastEmbed creates embeddings
  -> Qdrant stores vectors + metadata
  -> user asks question
  -> FastEmbed embeds question
  -> Qdrant retrieves relevant chunks
  -> Ollama generates answer locally
  -> Streamlit displays answer + citations
```

## Local Setup

Create `.env`:

```powershell
copy .env.example .env
```

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Qdrant in Docker:

```powershell
docker compose up -d qdrant
```

Make sure Ollama is running locally on Windows and has Mistral available:

```powershell
ollama list
```

Run the app:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Qdrant With Docker

This project uses Docker only for Qdrant. Ollama stays outside Docker and uses your existing local model cache.

```powershell
docker compose up -d qdrant
```

Then run Streamlit locally:

```powershell
streamlit run app.py
```

## Notes

- Qdrant must be running in Docker before indexing documents.
- FastEmbed may download its embedding model the first time it is used unless already cached.
- Ollama is contacted only through your local Windows server at `OLLAMA_BASE_URL`.
- Scanned PDFs need OCR before upload because `pypdf` extracts embedded text, not images.
