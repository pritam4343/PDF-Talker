# PDF Talker — Local RAG Document Assistant

A local-first PDF question-answering application built with Streamlit, PyMuPDF, Sentence Transformers, FAISS, BM25 and Ollama. It uses hybrid retrieval and asks the LLM to answer only from retrieved passages, with source labels and page-level provenance.

## Features

- Upload and index multiple text-based PDFs in a Streamlit session.
- Extract page text and retain source filename/page metadata.
- Chunk text with configurable word windows and overlap.
- Dense vector search (FAISS) plus BM25 keyword retrieval, fused with Reciprocal Rank Fusion.
- Local answer generation through Ollama.
- Display retrieved evidence passages with filenames and page numbers.
- Basic automated tests for chunking and prompt construction.

## Model configuration

- LLM: `qwen2.5:1.5b` through Ollama. The model is downloaded separately and runs locally.
- Embeddings: `BAAI/bge-small-en-v1.5` through Sentence Transformers.
- FAISS uses a CPU index. No paid API is used.

Model weights and Python dependencies are downloaded during initial setup, so first-time setup requires internet. After downloads are complete, normal inference can run locally. Verify each model's current licence and use terms before redistribution or commercial use.

## Requirements

- Python 3.10 or 3.11 recommended.
- 8 GB RAM minimum for a small collection; CPU indexing can be slow.
- Ollama installed locally: https://ollama.com/

## Setup (Windows PowerShell)

```powershell
cd pdf-talker
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:1.5b
ollama run qwen2.5:1.5b
```

Leave Ollama running, open a second terminal, activate the environment, and launch the UI:

```powershell
cd pdf-talker
.\.venv\Scripts\Activate.ps1
streamlit run app/main.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Setup (macOS/Linux)

```bash
cd pdf-talker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:1.5b
ollama run qwen2.5:1.5b
```

In a second terminal:

```bash
cd pdf-talker
source .venv/bin/activate
streamlit run app/main.py
```

## Usage

1. Open the Streamlit URL.
2. Upload one or more text-based PDFs.
3. Click **Build / rebuild document index** and wait for indexing to finish.
4. Ask a question in the chat input.
5. Inspect the retrieved source passages and confirm the answer against the cited pages.

If a PDF is a scanned image with no embedded text, this version will report that no text was extracted. OCR is not yet included.

## Run tests

```bash
python -m pytest
```

## Architecture

```text
PDFs -> PyMuPDF extraction -> page-aware chunks -> BGE embeddings -> FAISS
                                                     \-> BM25 keyword index
Question -> dense + keyword retrieval -> Reciprocal Rank Fusion -> evidence prompt
          -> local Qwen through Ollama -> answer + retrieved source passages
```

## Accuracy evaluation plan

Do not claim an accuracy percentage without measurement. Build a test set of 50–100 questions with manually verified answers and source pages. Include factual lookup, multi-page questions, exact identifiers, unanswerable questions, conflicting sources, and follow-up questions. Track:

- Retrieval Recall@k: whether a gold supporting passage appears in the top-k.
- Answer correctness: manual or rubric-based scoring against the reference answer.
- Faithfulness: whether answer claims are supported by retrieved evidence.
- Citation accuracy: whether cited source pages support the corresponding claims.
- Latency: retrieval time and end-to-end answer time.

Compare dense-only retrieval with hybrid retrieval and tune chunk size, overlap, top-k and prompt format. Record the test-set version and model versions with each run.

## Known limitations and next improvements

- OCR for scanned PDFs is not included.
- Indexes are session-memory only and are rebuilt after a restart.
- The current implementation is a single-user local prototype, not a hardened multi-user service.
- Retrieved passages are shown with provenance; model-generated source labels still require validation.
- Add automated citation verification, OCR, persistent index storage, per-user authentication, structured logging, and a FastAPI service before production use.
- Small local models may struggle with long-context synthesis or complex calculations. Validate high-impact answers against the source documents.

## Security and privacy notes

The app does not intentionally send PDF content to a hosted LLM API; generation is sent to the local Ollama endpoint. Initial model and package downloads require network access. Keep Ollama bound to localhost and do not expose it to untrusted networks. Treat document content as untrusted input.
