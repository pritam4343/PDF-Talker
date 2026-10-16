from __future__ import annotations
import streamlit as st
import requests
from rag import extract_pdf, chunk_pages, HybridRetriever, generate_answer, DEFAULT_LLM

st.set_page_config(page_title="PDF Talker", page_icon="📄", layout="wide")
st.title("📄 PDF Talker")
st.caption("Local RAG document assistant · PDF extraction · Hybrid search · Evidence-based answers")

with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input("Ollama model", value=DEFAULT_LLM, help="Pull this model with: ollama pull qwen2.5:1.5b")
    chunk_words = st.slider("Chunk size (words)", 200, 700, 400, 50)
    overlap_words = st.slider("Chunk overlap (words)", 0, 150, 60, 10)
    top_k = st.slider("Final passages", 2, 8, 4)
    st.markdown("---")
    st.caption("Documents and chat are kept in this Streamlit session. Model inference and embeddings run locally after setup.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []

st.subheader("1. Upload documents")
uploads = st.file_uploader("Choose one or more PDF files", type=["pdf"], accept_multiple_files=True)
col_a, col_b = st.columns([1, 1])
with col_a:
    build_clicked = st.button("Build / rebuild document index", type="primary", use_container_width=True)
with col_b:
    clear_clicked = st.button("Clear session", use_container_width=True)

if clear_clicked:
    for key in ("retriever", "chunks", "chat", "indexed_names"):
        st.session_state.pop(key, None)
    st.rerun()

if build_clicked:
    if not uploads:
        st.warning("Upload at least one PDF first.")
    else:
        try:
            all_pages = []
            names = []
            with st.status("Reading and indexing PDFs…", expanded=True) as status:
                for file in uploads:
                    if file.size > 25 * 1024 * 1024:
                        st.warning(f"Skipped {file.name}: file exceeds the 25 MB per-file limit.")
                        continue
                    try:
                        pages = extract_pdf(file.getvalue(), file.name)
                        if pages:
                            all_pages.extend(pages)
                            names.append(file.name)
                        else:
                            st.warning(f"No extractable text in {file.name}. It may be scanned; OCR is not enabled in this version.")
                    except Exception as exc:
                        st.warning(f"Could not process {file.name}: {exc}")
                chunks = chunk_pages(all_pages, chunk_words=chunk_words, overlap_words=overlap_words)
                if not chunks:
                    raise ValueError("No text chunks could be created from the uploaded files.")
                st.write(f"Created {len(chunks)} chunks from {len(names)} PDF(s). Loading embedding model and building search index…")
                retriever = HybridRetriever()
                retriever.build(chunks)
                st.session_state.retriever = retriever
                st.session_state.chunks = chunks
                st.session_state.indexed_names = names
                st.session_state.chat = []
                status.update(label="Index ready", state="complete", expanded=False)
            st.success(f"Indexed {len(chunks)} chunks from: {', '.join(names)}")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

if st.session_state.retriever:
    st.subheader("2. Ask questions")
    st.write("Indexed documents: " + ", ".join(st.session_state.indexed_names))
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Retrieved source passages"):
                    for source in msg["sources"]:
                        st.markdown(f"**{source['label']} — {source['source']}, page {source['page']}**")
                        st.write(source["text"])
    question = st.chat_input("Ask a question about your PDFs…")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating a local answer…"):
                    retrieved = st.session_state.retriever.retrieve(question, top_k=top_k, candidates=max(10, top_k * 2))
                    history = [m for m in st.session_state.chat[:-1] if m["role"] in ("user", "assistant")]
                    answer = generate_answer(question, retrieved, model=llm_model, history=history)
                    # Show provenance even if the model omits citations; do not fabricate claim-to-source mapping.
                    sources = [{"label": f"S{i}", "source": c.source, "page": c.page, "text": c.text}
                               for i, (c, _) in enumerate(retrieved, start=1)]
                st.markdown(answer)
                with st.expander("Retrieved source passages"):
                    for source in sources:
                        st.markdown(f"**{source['label']} — {source['source']}, page {source['page']}**")
                        st.write(source["text"])
                st.session_state.chat.append({"role": "assistant", "content": answer, "sources": sources})
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Ollama. Start Ollama and run `ollama pull qwen2.5:1.5b`.")
            except requests.exceptions.HTTPError as exc:
                st.error(f"Ollama returned an error: {exc}. Check that the selected model has been downloaded.")
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
else:
    st.info("Upload PDFs and build the index to start asking questions.")

with st.expander("Troubleshooting and limitations"):
    st.markdown("""
- **Scanned PDFs:** this version extracts embedded text only; OCR is not included.
- **First launch:** the embedding model must be downloaded once and may take time on CPU.
- **RAM:** close other memory-heavy applications. Embedding large collections may take time on 8 GB RAM.
- **Accuracy:** verify important answers against the cited pages. RAG reduces but does not eliminate hallucinations.
- **Privacy:** processing is local, but Ollama and model downloads need internet during initial setup. Do not expose the local service to untrusted networks.
- **Session storage:** uploaded files and the index live in the Streamlit session memory; they are not retained after the session is cleared or the app restarts.
""")
