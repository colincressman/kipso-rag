# Private Document Intelligence Platform

> Fully local, on-premises document intelligence — grounded Q&A, structured corpus extraction, and large-document summarization over private technical libraries. No API keys. No cloud. No data leaves your hardware.

![Chat UI — streaming RAG response with inline citations](docs/screenshots/RAG-Chat.png)

---

## Features

- **Private by design** — runs entirely on your hardware; PostgreSQL + pgvector + Ollama, no external API calls
- **Hybrid retrieval** — BM25 + pgvector ANN, cross-encoder reranking, HyDE query expansion, LLM contextual compression, and RRF merging
- **Intent-aware routing** — DeBERTa-v3-large NLI zero-shot classifier automatically routes queries to corpus, web search, or conversation history
- **Branched document extraction** — extract user-defined information types across a multi-document corpus via a two-pass scan+synthesis pipeline; handles corpora larger than any context window
- **Map-reduce summarization** — summarize arbitrarily large documents via an async job queue with no context window limits
- **Grounded answers** — inline `[cNNNNNN]` citations, lexical coverage scoring, and entity faithfulness checks on every answer
- **Distributed inference** — optional satellite service offloads DeBERTa, cross-encoder, and HyDE model to a secondary GPU

---

## Getting Started

### Windows (recommended)

Run `rag_installer.exe`. The setup wizard handles PostgreSQL 16 + pgvector, Ollama, model downloads, directory creation, and config setup. After installation, use the GUI launcher (`launcher/app.py`) to start/stop the server, monitor VRAM, and adjust settings.

![Installer setup wizard](docs/screenshots/RAG-Installer.png)

![GUI launcher — server status, VRAM monitor, and console](docs/screenshots/RAG-Console.png)

### Manual (Linux / macOS / developer)

**Prerequisites**

| Dependency | Notes |
|---|---|
| Python 3.11+ | Tested on 3.13 |
| [Ollama](https://ollama.com) | Running at `http://localhost:11434` |
| PostgreSQL 16 + pgvector | DSN: `postgresql://postgres:postgres@localhost/rag` |
| NVIDIA GPU (recommended) | 8 GB VRAM minimum; CPU inference works at reduced speed |

```bash
# 1. Pull Ollama models
ollama pull qwen3-embedding:latest
ollama pull qwen3.5:9b

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the server
python main.py serve

# 4. Open http://localhost:8000
```

### Optional: Satellite Inference Service

A secondary machine running `inference_service/server.py` offloads three models to preserve VRAM on the primary GPU. HuggingFace models download automatically on first use.

| Model | Role |
|---|---|
| `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | NLI intent classifier (auto-downloads via HuggingFace) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker (auto-downloads via HuggingFace) |
| 0.8B Ollama model | HyDE hypothetical document generation |

```bash
# On the satellite machine
pip install -r inference_service/requirements.txt
python inference_service/server.py
```

Without the satellite, all three models run locally: DeBERTa and the cross-encoder load via HuggingFace/sentence-transformers (auto-downloaded on first use), and HyDE falls back to the main Ollama LLM. The satellite is a VRAM optimization — it offloads these models to keep the main GPU free for generation. The primary machine auto-discovers the satellite via subnet scan on port 8100.

---

## Q&A

Ingest documents and ask questions with grounded, cited answers.

```bash
python main.py ingest path/to/document.pdf
python main.py query "What is the maximum operating temperature?"
# or open http://localhost:8000
```

![Chat UI — streaming response with routing badge and inline citations](docs/screenshots/RAG-Chat.png)

The retrieval pipeline: acronym expansion → intent classification → HyDE query expansion → BM25 + pgvector hybrid search → cross-encoder reranking → LLM contextual compression → grounded answer with inline `[cNNNNNN]` citations.

**Auto-routing** — queries are classified as corpus, web search, or conversational. Web search (Wikipedia → Bing RSS → DuckDuckGo cascade with 24h SQLite cache) fires automatically for live or out-of-corpus queries. Override with Force-RAG / Force-Web buttons in the UI.

**Collections** — organize documents into named groups and scope queries naturally:

```bash
python scripts/ops/manage_collections.py create "manuals" "Equipment Manuals"
python main.py ingest pump_manual.pdf --collection manuals
# Natural-language scoping works automatically in queries:
# "In the manuals, what is the torque spec for the shaft coupling?"
```

**Library** (`http://localhost:8000/library.html`) — document browser with collection tree sidebar, drag-to-assign, and mobile-responsive layout.

![Library — collection tree with documents assigned](docs/screenshots/RAG-Library.png)

---

## Branched Document Extraction

Extract structured information across an entire document corpus — designed for corpora too large to fit in any context window.

Open `http://localhost:8000/extraction.html` for the UI, or run via CLI/API.

Every extracted item is **verbatim text from the source documents** — the LLM selects, never synthesizes. No hallucinations; every item is a direct quote with a page citation.

**How it works:**

1. Define a **Project** — a name and a set of source documents already ingested into the corpus.
2. Define **Branches** — each branch is a named topic to find across the corpus (e.g. "risk factors", "API contracts", "operating limits", "definitions"). Use `keyword` mode for exact-match topics or `semantic` mode for concept-based discovery. Mix modes freely across branches.
3. Each branch independently retrieves candidates from the shared corpus, then runs **LLM selection in as many batches as the corpus requires**:
   - Coarse pass over chunk previews (IDs + short excerpts) — the LLM identifies relevant candidates. Minimal token cost; scales to any corpus size.
   - Fine pass over the full text of shortlisted chunks — the LLM makes the final selection of chunks to include verbatim.
4. **Post-branch passes** — optional report-building LLM passes that run after all branches finish. Each pass is now explicit about:
   - **Input Source** — `All branch items`, `Selected branch items`, or `Previous pass output`
   - **Execution Mode** — `Single pass`, `Per-branch`, `Map-reduce`, or `Chain from previous output`
   - Use **Single pass** when the extracted items are small enough to fit comfortably in one call.
   - Use **Per-branch** when you want one mini-summary per branch and then join those summaries together.
   - Use **Map-reduce** when the extracted items are large: the system runs the same prompt over multiple batches, then performs a **true final combine step** to merge the partial outputs into one polished section.
   - Use **Chain from previous output** when a later pass should transform or refine the result of an earlier pass instead of re-reading the branch items.
   - Example: Pass 1 can summarize **all branch items** into a proposal-readiness report, and Pass 2 can **chain from previous output** to turn that report into an executive summary or bid/no-bid recommendation.
5. **Cross-branch deduplication** — items appearing in multiple branches are flagged with `⚠`.
6. Output: a structured **Markdown report** with a TOC, per-branch sections, page citations, and a branch stats table.

![Extraction report — TOC, branch sections, and cross-reference flags](docs/screenshots/RAG-Extraction.png)


**AI branch suggestions** — describe your document type and the system proposes a pre-filled branch list for you to review before running.

```python
from extraction.project_runner import run_project
from extraction.branch_config import ProjectConfig

result = run_project(ProjectConfig.from_dict({...}), emit=print)
# result.report_path    — path to the generated Markdown report
# result.branch_results — per-branch lists of verbatim ExtractionItems
# result.post_pass_results — optional report-building pass outputs
```

---

## Document Summarization

Map-reduce summarization over arbitrarily large documents via an async job queue.

- Documents ≤ 60K chars: single-pass summary
- Larger documents: segmented into 15K-char batches → per-segment summaries → final reduce pass
- Jobs run in a background worker; no HTTP timeout risk

```bash
python scripts/maintenance/generate_summaries.py           # all unsummarized documents
python scripts/maintenance/generate_summaries.py --force   # regenerate all
python scripts/maintenance/generate_summaries.py --doc-id 277ac35c
```

Summaries are stored in PostgreSQL and automatically used for `summary`-intent RAG queries.

![Book Summarizer — document cards, page-range summaries, and full-document job queue](docs/screenshots/RAG-Summary.png)

---

## Programmatic API

```python
from api import rag_retrieve, llm_answer

result = rag_retrieve("What is the maximum operating temperature?", top_k=8)
# result["hits"]          — list of retrieved chunks with scores
# result["routing"]       — intent, source_type_filter, strategy used
# result["context_pack"]  — packed context string

answer = llm_answer("What is the maximum operating temperature?", result)
# answer["answer"]    — generated text with inline citations
# answer["citations"] — list of cited chunk_ids
# answer["mode"]      — high_confidence / medium / low / no_coverage
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Primary Machine                    │
│                                                     │
│  FastAPI server (:8000)                             │
│    ├─ Chat / Q&A        → retrieval + LLM answer    │
│    ├─ Extraction Studio → branch pipeline           │
│    └─ Summarization     → async job worker          │
│                                                     │
│  PostgreSQL 16 + pgvector  (chunks, docs, jobs)     │
│  Ollama  qwen3-embedding:latest  +  rag-llm         │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (auto-discovered on :8100)
┌──────────────────────▼──────────────────────────────┐
│         Satellite Inference Service (optional)      │
│                                                     │
│  DeBERTa-v3-large   — intent classification         │
│  ms-marco-MiniLM    — cross-encoder reranking       │
│  0.8B Ollama model  — HyDE query expansion          │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

| File | Controls |
|---|---|
| `configs/runtime.yaml` | DB DSN, embedding model, retrieval thresholds, HyDE, chunking, conversation retention |
| `configs/llm.yaml` | LLM model, prompt limits, confidence thresholds, summarization parameters |
| `configs/scoring.yaml` | Reranking weights (grid-validated), CDI threshold, NLI blend ratios |
| `configs/parsing.yaml` | PDF page classifier thresholds, Marker extraction settings |
| `configs/extraction.yaml` | Batch sizes, scan/synthesis temperatures, deduplication thresholds |

---

## Evaluation

Retrieval harness (50 cases, 6 intent types, no LLM calls):

| Metric | Score |
|---|---|
| R@10 | 0.928 |
| MRR | 0.989 |
| NDCG | 0.936 |
| MAP | 0.886 |

Golden Q&A evaluation (25 grounded questions, end-to-end with LLM): **22/25 (88%)** with `qwen3.5:9b`.

```bash
python scripts/eval/recall_mrr_harness.py --dataset data/qa/retrieval_recall_dataset.json --save-history
python scripts/eval/run_golden_eval.py --output data/diagnostics/eval.json
```

---

## Development

```bash
# Fast suite — no LLM calls, ~90s, 679 tests
.venv\Scripts\python.exe -m pytest tests/ -q \
  --ignore=tests/test_llm.py \
  --ignore=tests/test_retrieval.py \
  --ignore=tests/test_question_battery.py

# Full suite — requires Ollama, ~20min
.venv\Scripts\python.exe -m pytest tests/ -x -q
```

---

## Project Layout

```
api.py                  Public API: rag_retrieve(), llm_answer()
main.py                 CLI: serve | ingest | query | status
services/               Thin facades decoupling routes from pipeline internals
pipeline/
  ingest_v3.py          Unified ingest (PDF/DOCX/text/code/markdown)
  extract/              Page classifier, Marker, pdfplumber, math extractor
  normalize/            OCR artifact cleanup
  structure/            Heading hierarchy parser + structural role assignment
  chunk/                Splitting strategies + Chunk dataclass
  embed/                OllamaEmbedder + _TestEmbedder factory
retrieval/
  query.py              Main orchestrator (15+ sub-modules)
  router.py             Intent router (DeBERTa NLI + cosine blend)
  hyde.py               HyDE + step-back query expansion
  rerank.py             Hybrid scoring (grid-validated weights)
  cross_encoder.py      Cross-encoder rescore
  context_compress.py   LLM-based contextual compression
  internet_fallback.py  Multi-provider web search + SQLite TTL cache
llm/
  answer.py             Answer orchestrator
  summarize.py          Single-pass + map-reduce summarization
  grounding.py          Lexical coverage + entity faithfulness
  citations.py          Citation normalization + inline enforcement
db/                     PostgreSQL: documents, chunks, collections,
                          conversations, jobs queue, chunk_questions
server/
  app.py                FastAPI app with lifespan startup
  job_worker.py         Async summarize job dispatcher
  routes/               chat, library, extraction, summarize, admin
extraction/             Branched extraction: two-pass scan+synthesis,
                          dedup, report builder, AI branch suggestions
inference_service/
  server.py             Satellite: /classify_intent + /rerank + /health
launcher/
  app.py                Windows GUI launcher (VRAM monitor + config UI)
installer/
  installer.py          Windows setup wizard (PostgreSQL + Ollama + models)
configs/                runtime.yaml, llm.yaml, scoring.yaml, parsing.yaml,
                          models.yaml, extraction.yaml
scripts/
  eval/                 Golden eval, retrieval harness, grid search
  ops/                  query_cli, ingest_one, db_status, read_trace
  maintenance/          rebuild_corpus_manifest, reset_db, rag_diagnostics
tests/                  679 passing tests
```

---

## License

MIT License — Copyright (c) 2026 Colin Cressman. See [LICENSE](LICENSE) for the full text.
