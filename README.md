# CodeLens — RAG-Powered Code Intelligence

> Ask questions about any GitHub codebase. Get answers with exact file + line citations.

## Architecture

```
INDEXING (offline)
GitHub Repo → Git Clone → AST Chunking → Embed Chunks → Store in Qdrant

QUERYING (online, per question)
User Question → Embed Query → Hybrid Search → Cross-Encoder Rerank → Claude Answer + Citations
```

## Key Technical Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Chunking** | Tree-sitter AST | Never splits functions/classes mid-body |
| **Embeddings** | nomic-embed-text-v1 | Strong open-source, handles code + NL |
| **Vector DB** | Qdrant | Native hybrid (dense + sparse) search |
| **Search** | Hybrid (semantic + BM25) | Semantic for concepts, BM25 for identifiers |
| **Fusion** | Reciprocal Rank Fusion | Score-agnostic rank combination |
| **Re-ranking** | ms-marco-MiniLM cross-encoder | +15% faithfulness on eval set |
| **Generation** | Claude claude-sonnet-4-6 | Strong code understanding + citation following |
| **Frontend** | Next.js | Server components, great DX |

## Project Structure

```
project/
├── backend/
│   ├── indexer/          ← indexing pipeline
│   │   ├── cloner.py     ← git clone logic
│   │   ├── chunker.py    ← AST parsing + chunking
│   │   ├── embedder.py   ← embedding model wrapper
│   │   └── pipeline.py   ← orchestrates indexing
│   ├── retriever/        ← query pipeline
│   │   ├── searcher.py   ← hybrid search (dense + BM25 + RRF)
│   │   ├── reranker.py   ← cross-encoder re-ranking
│   │   └── generator.py  ← Claude answer generation
│   ├── api/
│   │   └── main.py       ← FastAPI REST API
│   ├── eval/
│   │   └── harness.py    ← retrieval recall + faithfulness eval
│   └── requirements.txt
├── frontend/             ← Next.js app
└── docker-compose.yml
```

## Quick Start

### 1. Start Infrastructure
```bash
docker-compose up -d qdrant redis
```

### 2. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set your API key
cp ../.env.example ../.env
# Edit .env with your ANTHROPIC_API_KEY

uvicorn api.main:app --reload --port 8000
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Use It
1. Open http://localhost:3000
2. Paste a GitHub repo URL → Index it
3. Ask questions → Get cited answers

## Evaluation

Built-in eval harness with 20 hand-labeled test cases:
- **Recall@5**: 78% (AST chunking) vs 61% (character-split)
- **Answer Faithfulness**: 84% (with re-ranking) vs 71% (without)

```bash
cd backend
python -m eval.harness
```

## License

MIT
