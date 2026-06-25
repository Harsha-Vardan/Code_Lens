"""
FastAPI REST API for CodeLens.

Two main endpoints matching the two-phase architecture:
1. POST /api/index  — starts a background indexing job (offline phase)
2. POST /api/query  — runs the full query pipeline (online phase)

Additional endpoints:
- GET  /api/index/{job_id}  — poll indexing job status
- GET  /api/repos           — list indexed repositories
- GET  /api/health          — health check

Design decisions:
- Background tasks for indexing (non-blocking, repo clone + embed is slow)
- In-memory job store (use Redis in production for persistence)
- CORS enabled for Next.js frontend at localhost:3000
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Lazy imports to avoid loading heavy ML models at import time
# They're loaded on first use instead

app = FastAPI(
    title="CodeLens API",
    description="RAG-powered code intelligence — ask questions about any codebase",
    version="1.0.0",
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    """Request to start indexing a GitHub repository."""
    github_url: str = Field(
        ..., 
        description="Full GitHub URL (HTTPS)",
        examples=["https://github.com/expressjs/express"]
    )

class QueryRequest(BaseModel):
    """Request to query an indexed codebase."""
    repo_id: str = Field(
        ..., 
        description="Repository identifier (from indexing response)"
    )
    question: str = Field(
        ..., 
        description="Natural language question about the codebase"
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=20,
        description="Number of source chunks to retrieve"
    )

class IndexStatusResponse(BaseModel):
    """Status of an indexing job."""
    job_id: str
    status: str  # queued, cloning, indexing, complete, error
    progress: int = 0
    repo_id: Optional[str] = None
    total_chunks: Optional[int] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-Memory State
# ---------------------------------------------------------------------------

# In production, use Redis for job state persistence across restarts
jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Indexing Background Task
# ---------------------------------------------------------------------------

def run_indexing(job_id: str, github_url: str, repo_id: str):
    """
    Background task that runs the full indexing pipeline.
    
    Steps:
    1. Clone the repo (shallow)
    2. Create Qdrant collection
    3. Walk files → AST chunk → embed → upsert
    4. Save vocabulary
    
    Updates job status at each step so the frontend can poll progress.
    """
    try:
        # Import here to avoid loading models at app startup
        from backend.indexer.cloner import RepoCloner
        from backend.indexer.pipeline import IndexingPipeline
        
        jobs[job_id]["status"] = "cloning"
        jobs[job_id]["progress"] = 10
        
        # Clone the repository
        cloner = RepoCloner()
        clone_path, actual_repo_id = cloner.clone(
            github_url, 
            repo_id=repo_id,
            force_reclone=True
        )
        
        jobs[job_id]["status"] = "indexing"
        jobs[job_id]["progress"] = 20
        
        # Initialize the indexing pipeline
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        pipeline = IndexingPipeline(qdrant_url=qdrant_url)
        pipeline.create_collection()
        
        # Run the full pipeline with progress updates
        def progress_callback(status: str, progress: int):
            # Scale pipeline progress (0-100) to our range (20-95)
            scaled = 20 + int(progress * 0.75)
            jobs[job_id]["status"] = "indexing"
            jobs[job_id]["progress"] = min(scaled, 95)
            jobs[job_id]["detail"] = status
        
        total_chunks = pipeline.index_repo(
            clone_path, 
            repo_id,
            progress_callback=progress_callback
        )
        
        # Done!
        jobs[job_id] = {
            "status": "complete",
            "progress": 100,
            "repo_id": repo_id,
            "total_chunks": total_chunks,
        }
        
    except Exception as e:
        jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "repo_id": repo_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "codelens-api"}


@app.post("/api/index", response_model=IndexStatusResponse)
async def start_indexing(
    req: IndexRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start indexing a GitHub repository.
    
    Returns a job_id that can be used to poll status via GET /api/index/{job_id}.
    Indexing runs in the background — the clone + embed process can take minutes
    for large repos.
    """
    job_id = str(uuid.uuid4())
    
    # Extract repo ID from URL
    repo_id = req.github_url.rstrip('/').split('/')[-1]
    if repo_id.endswith('.git'):
        repo_id = repo_id[:-4]
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued", 
        "progress": 0, 
        "repo_id": repo_id
    }
    
    # Launch indexing in the background
    background_tasks.add_task(run_indexing, job_id, req.github_url, repo_id)
    
    return IndexStatusResponse(
        job_id=job_id, 
        repo_id=repo_id, 
        status="queued",
        progress=0
    )


@app.get("/api/index/{job_id}", response_model=IndexStatusResponse)
async def get_job_status(job_id: str):
    """
    Poll the status of an indexing job.
    
    Frontend should poll this every 2-3 seconds until status is
    'complete' or 'error'.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return IndexStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        repo_id=job.get("repo_id"),
        total_chunks=job.get("total_chunks"),
        error=job.get("error"),
    )


@app.post("/api/query")
async def query_codebase(req: QueryRequest):
    """
    Query an indexed codebase with a natural language question.
    
    Full pipeline:
    1. Hybrid search (dense + BM25 + RRF) → top 20 candidates
    2. Cross-encoder re-ranking → top_k results
    3. Claude answer generation with citations
    
    Returns answer text + structured citation objects.
    """
    # Import here to avoid loading models at app startup
    from backend.retriever.searcher import HybridSearcher
    from backend.retriever.reranker import Reranker
    from backend.retriever.generator import Generator
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="ANTHROPIC_API_KEY not configured"
        )
    
    try:
        # Step 1: Hybrid search — retrieve top 20 candidates
        searcher = HybridSearcher(qdrant_url, "codelens_chunks")
        candidates = searcher.search(
            req.question, 
            req.repo_id, 
            top_k=20
        )
        
        if not candidates:
            return {
                "answer": "No relevant code chunks found for your question. "
                          "The repository may not be indexed yet, or the question "
                          "may not match any indexed content.",
                "citations": [],
                "model_used": "none",
                "chunks_retrieved": 0,
                "query": req.question,
            }
        
        # Step 2: Cross-encoder re-ranking — pick top_k from top 20
        reranker = Reranker()
        top_chunks = reranker.rerank(
            req.question, 
            candidates, 
            top_k=req.top_k
        )
        
        # Step 3: Generate answer with Claude
        generator = Generator(api_key=api_key)
        result = generator.generate(req.question, top_chunks)
        
        return result
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Repository '{req.repo_id}' not found. "
                   f"Please index it first via POST /api/index"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Query failed: {str(e)}"
        )


@app.get("/api/repos")
async def list_repos():
    """
    List all indexed repositories by checking stored vocabulary files.
    
    A repo is considered indexed if it has a vocabulary file in the
    metadata directory.
    """
    from backend.indexer.pipeline import METADATA_DIR
    
    repos = []
    if os.path.exists(METADATA_DIR):
        for filename in os.listdir(METADATA_DIR):
            if filename.endswith("_vocab.json"):
                repo_id = filename.replace("_vocab.json", "")
                repos.append({"repo_id": repo_id, "status": "indexed"})
    
    return {"repos": repos}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
