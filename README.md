# FastAPI RAG Copilot

A retrieval-augmented generation system that answers questions from the FastAPI documentation with inline citations, confidence scoring, and graceful fallback handling.

## Architecture

```
ingestion/          Scrape, clean, chunk, and index the FastAPI docs
retrieval/          Dense (Qdrant) + sparse (BM25) retrieval, RRF fusion, cross-encoder reranking
generation/         LLM prompting, citation verification, confidence scoring, fallback
evaluation/         Golden Q&A set, metrics harness, Streamlit dashboard
```

## Pipeline

```
User question
    |
    v
Dense retriever (Qdrant, cosine, all-MiniLM-L6-v2)
    +
Sparse retriever (BM25Okapi)
    |
    v
RRF fusion (weighted reciprocal rank fusion)
    |
    v
Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
    |
    v
Confidence gate (threshold = 0.45)
    |
    +-- above threshold --> LLM generation (Llama 3.3 70B via Groq)
    |                           |
    |                           v
    |                       Citation verification
    |                           |
    |                           v
    |                       Confidence scoring
    |
    +-- below threshold --> Fallback response (closest sections + explanation)
```

## Setup

```bash
git clone <repo-url>
cd coding-copilot
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Add your Groq API key (free at https://console.groq.com/keys):
```
GROQ_API_KEY=gsk_...
```

## Ingestion

Scrape the FastAPI docs, clean, chunk, and build indexes:
```bash
python -m ingestion.ingest --source corpus/ --rebuild
```

This creates:
- `corpus/` -- raw HTML pages
- `corpus_clean/` -- cleaned content
- `chunks/` -- JSONL chunk files
- `indexes/` -- Qdrant vectors + BM25 index + metadata

## Usage

### CLI
```bash
python -m generation.llm "How do I add CORS middleware?"
```

### Dashboard
```bash
streamlit run evaluation/dashboard.py
```

### Evaluation
```bash
python eval.py --strategy hybrid
python eval.py --strategy dense-only
```

## Project structure

```
contract.py                 Pydantic I/O schema (QueryRequest, AnswerResponse, Citation)
eval.py                     Evaluation runner -- generates Markdown reports
.env.example                Template for API keys

ingestion/
    scrape_corpus.py        Fetch FastAPI docs (current + versioned snapshots)
    normalise.py            Strip navigation chrome from HTML
    chunker.py              Dual-strategy chunking (heading-based + fixed-size)
    indexer.py              Build Qdrant vector index + BM25 index
    ingest.py               CLI entry point for the full ingestion pipeline

retrieval/
    retriever.py            Dense (Qdrant) and sparse (BM25) retrievers
    fusion.py               Reciprocal Rank Fusion with configurable weights
    reranker.py             Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)

generation/
    prompt.py               System prompt and user prompt templates
    llm.py                  Groq API integration (Llama 3.3 70B)
    verifier.py             Citation parsing and rule-based verification
    confidence.py           Confidence scoring (retrieval + citation + completeness)
    fallback.py             Graceful refusal when retrieval confidence is low

evaluation/
    golden_qa.json          75 test questions across 5 categories
    metrics.py              Retrieval, answer, citation, and refusal metrics
    dashboard.py            Streamlit dashboard for interactive inspection
```

## Models used

| Component | Model | Size |
|-----------|-------|------|
| Embedding | all-MiniLM-L6-v2 | 22M params, 384-dim |
| Reranking | ms-marco-MiniLM-L-6-v2 | 22M params |
| Generation | Llama 3.3 70B (via Groq) | 70B params |

## Key design decisions

- **Dual chunking** -- heading-based chunks carry semantic structure; fixed-size chunks ensure full coverage
- **Hybrid retrieval** -- dense captures paraphrases, sparse captures exact API names and error codes
- **Pre-generation filtering** -- outdated chunks are removed before the LLM sees them; heading chunks preferred over arbitrary fixed windows
- **Citation enforcement** -- the contract model rejects responses with empty citations unless `unverified` is explicitly set
- **Confidence breakdown** -- three independent signals (retrieval quality, citation validity, answer completeness) weighted into a single score
