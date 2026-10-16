from app.rag import chunk_pages, build_prompt, Chunk


def test_chunk_pages_preserves_source_and_page():
    pages = [{"source": "sample.pdf", "page": 3, "text": "word " * 120}]
    chunks = chunk_pages(pages, chunk_words=50, overlap_words=10)
    assert len(chunks) >= 2
    assert all(c.source == "sample.pdf" and c.page == 3 for c in chunks)


def test_chunk_validation():
    try:
        chunk_pages([], chunk_words=50, overlap_words=50)
        assert False, "Expected invalid overlap to raise ValueError"
    except ValueError:
        pass


def test_prompt_includes_evidence_and_abstention_rule():
    chunk = Chunk(text="The project budget is £100.", source="budget.pdf", page=2, chunk_id=0)
    prompt = build_prompt("What is the budget?", [(chunk, 0.02)])
    assert "budget.pdf" in prompt
    assert "Page: 2" in prompt
    assert "insufficient evidence" in prompt.lower()
