import json
import sys
from pathlib import Path
import urllib.request
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.embed.embedder import _TestEmbedder, create_embedder, embed_chunks
from pipeline.embed.index import build_embedding_index, build_index_for_directory


def _chunks_payload() -> dict:
    return {
        "source_structured_path": "data/structured/spec.structured.json",
        "chunk_count": 2,
        "chunks": [
            {
                "chunk_id": "abc-c000000",
                "doc_id": "abc",
                "source_path": "data/markdown/spec.md",
                "section_id": "s1",
                "chunk_index_in_section": 0,
                "text": "Power unit shall support 24V input.",
                "token_count_est": 10,
                "word_count": 6,
                "title": "1 Scope",
                "path": ["1 Scope"],
                "path_text": "1 Scope",
                "level": 2,
                "page_start": 1,
                "page_end": 1,
                "has_table": False,
                "metadata": {},
            },
            {
                "chunk_id": "abc-c000001",
                "doc_id": "abc",
                "source_path": "data/markdown/spec.md",
                "section_id": "s2",
                "chunk_index_in_section": 0,
                "text": "Input current should remain below threshold.",
                "token_count_est": 12,
                "word_count": 6,
                "title": "2 Requirements",
                "path": ["2 Requirements"],
                "path_text": "2 Requirements",
                "level": 2,
                "page_start": 2,
                "page_end": 2,
                "has_table": False,
                "metadata": {},
            },
        ],
    }


def test_test_embedder_dimension_and_norm():
    """_TestEmbedder is the offline test-only backend; verify it produces normalised vectors."""
    emb = _TestEmbedder(dimension=128)
    vec = emb.embed_query("hello world")
    assert len(vec) == 128
    norm = sum(v * v for v in vec) ** 0.5
    assert round(norm, 6) in {0.0, 1.0}


def test_create_ollama_embedder_factory_only():
    emb = create_embedder("ollama", model_name="qwen3-embedding", ollama_base_url="http://localhost:11434")
    # Factory should return an object with embed methods without making network calls here
    assert hasattr(emb, "embed_texts")
    assert hasattr(emb, "embed_query")


@pytest.mark.skipif(
    "CI" in __import__("os").environ,
    reason="Skip live Ollama test in CI by default",
)
def test_ollama_embedder_live_smoke_optional():
    """Optional live test: runs only when Ollama server+model are available."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as _:
            pass
    except Exception:
        pytest.skip("Ollama server not reachable on localhost:11434")

    emb = create_embedder("ollama", model_name="qwen3-embedding", ollama_base_url="http://localhost:11434")
    vec = emb.embed_query("embedding smoke test")
    assert isinstance(vec, list)
    assert len(vec) > 100


def test_embed_chunks_adds_embedding_field():
    payload = _chunks_payload()
    emb = create_embedder("_test", dimension=96)
    rows = embed_chunks(payload["chunks"], embedder=emb)
    assert len(rows) == 2
    assert "embedding" in rows[0]
    assert len(rows[0]["embedding"]) == 96


def test_build_embedding_index_from_file(tmp_path: Path):
    chunks_path = tmp_path / "doc.chunks.merged.json"
    out_path = tmp_path / "doc.index.json"
    chunks_path.write_text(json.dumps(_chunks_payload()), encoding="utf-8")

    payload = build_embedding_index(
        str(chunks_path),
        output_path=str(out_path),
        backend="_test",
        dimension=64,
    )
    assert out_path.exists()
    assert payload["vector_count"] == 2
    assert payload["dimension"] == 64

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["vector_count"] == 2
    assert len(saved["items"][0]["embedding"]) == 64


def test_build_index_for_directory(tmp_path: Path):
    d = tmp_path / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.chunks.merged.json").write_text(json.dumps(_chunks_payload()), encoding="utf-8")
    (d / "b.chunks.merged.json").write_text(json.dumps(_chunks_payload()), encoding="utf-8")

    created = build_index_for_directory(str(d), backend="_test", dimension=32)
    assert len(created) == 2
    assert all(Path(p).exists() for p in created)


@pytest.mark.slow
def test_sentence_transformers_embedder_dimension_and_similarity():
    """Integration: load the real SentenceTransformers model and verify vectors."""
    from pipeline.embed.embedder import SentenceTransformersEmbedder
    emb = SentenceTransformersEmbedder()
    similar_a = emb.embed_query("decision tree recursive binary splitting")
    similar_b = emb.embed_query("recursive splitting of a binary decision tree")
    unrelated = emb.embed_query("chocolate cake recipe with frosting")

    assert len(similar_a) > 100, "expected high-dimensional embedding"
    # cosine similarity via dot product (vectors are L2-normalised)
    dot_similar = sum(a * b for a, b in zip(similar_a, similar_b))
    dot_unrelated = sum(a * b for a, b in zip(similar_a, unrelated))
    assert dot_similar > dot_unrelated, (
        f"similar pair ({dot_similar:.4f}) should score higher than unrelated pair ({dot_unrelated:.4f})"
    )


@pytest.mark.slow
def test_sentence_transformers_embed_texts_batch():
    """Integration: batch embed produces consistent results with embed_query."""
    from pipeline.embed.embedder import SentenceTransformersEmbedder
    emb = SentenceTransformersEmbedder()
    texts = ["neural network backpropagation", "gradient descent optimisation"]
    batch = emb.embed_texts(texts)
    single0 = emb.embed_query(texts[0])

    assert len(batch) == 2
    assert len(batch[0]) == len(single0)
    # Batch and single-query results must be essentially identical
    diff = max(abs(a - b) for a, b in zip(batch[0], single0))
    assert diff < 1e-4, f"batch vs single-query mismatch: max_diff={diff}"
