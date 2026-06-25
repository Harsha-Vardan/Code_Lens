"""
Indexing Pipeline — Orchestrates the complete indexing flow.

This module ties together cloning, chunking, embedding, and vector storage.
It's the "offline phase" of RAG — it runs once per repo (or when the repo
updates) and populates the vector database.

Flow:
    GitHub Repo → Clone → Walk Files → AST Chunk → Embed → Store in Qdrant

The pipeline creates a Qdrant collection with DUAL vector support:
- Dense vectors: from the embedding model, capture semantic meaning
- Sparse vectors: BM25 term frequencies, capture keyword overlap

This dual-vector setup is what enables hybrid search at query time.

Interview: "Why store the BM25 vocabulary separately?"
    "BM25 sparse vectors need a shared vocabulary — the query vector and
    document vectors must use the same term-to-index mapping so they can
    be compared. I store the vocabulary built during indexing and load it
    at query time to convert the user's query into the same sparse vector space."
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector
)
import numpy as np
import json
import re
import os
from pathlib import Path
from typing import Optional
from collections import Counter

from backend.indexer.chunker import chunk_file, chunk_file_with_fallback, should_skip_path, Chunk
from backend.indexer.embedder import Embedder
from backend.indexer.cloner import RepoCloner


# Directory to store vocabulary files and indexing metadata
METADATA_DIR = os.path.join(os.path.expanduser("~"), ".codelens", "metadata")


class IndexingPipeline:
    """
    Orchestrates the full indexing pipeline for a code repository.
    
    Responsibilities:
    1. Create/recreate Qdrant collection with dual vector config
    2. Walk source files and parse them into AST chunks
    3. Build BM25 vocabulary across all chunks
    4. Batch-embed all chunks (dense vectors)
    5. Compute BM25 sparse vectors
    6. Upsert everything into Qdrant
    7. Persist vocabulary for query-time use
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """
        Args:
            qdrant_url: URL of the Qdrant instance
        """
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = Embedder()
        self.collection_name = "codelens_chunks"
        self.vector_size = self.embedder.dimension  # dynamically from model
        os.makedirs(METADATA_DIR, exist_ok=True)
    
    def create_collection(self):
        """
        Create a Qdrant collection that supports BOTH dense and sparse vectors.
        
        This is what enables hybrid search — one collection, two vector types:
        - "dense": embedding vectors for semantic similarity search
        - "sparse": BM25 term frequency vectors for keyword search
        
        Using recreate_collection drops the old collection if it exists.
        In production, you'd want incremental updates instead.
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                # Dense vector: from embedding model, captures semantic meaning
                "dense": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                # Sparse vector: BM25 term frequencies, captures keyword overlap
                # On-disk=False because sparse vectors are small and we want speed
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )
        print(f"Created collection '{self.collection_name}' "
              f"(dense: {self.vector_size}d, sparse: BM25)")
    
    def tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer for BM25. Splits on non-alphanumeric characters,
        lowercases everything.
        
        For code, this handles camelCase and snake_case reasonably —
        'verifyToken' becomes ['verifytoken'] and 'verify_token' becomes
        ['verify', 'token']. Not perfect but good enough for BM25.
        
        A production system would use a code-aware tokenizer that handles
        camelCase splitting, but rank-bm25 works surprisingly well with
        this simple approach.
        """
        return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
    
    def compute_bm25_vector(
        self, 
        tokens: list[str], 
        vocab: dict[str, int]
    ) -> SparseVector:
        """
        Convert tokenized text to a sparse vector using the shared vocabulary.
        
        Qdrant represents sparse vectors as {indices: [...], values: [...]}
        where indices are vocabulary positions and values are term frequencies.
        Only non-zero entries are stored — that's what makes it "sparse."
        
        For a document with 50 unique terms out of a 10,000-term vocabulary,
        only 50 entries are stored instead of 10,000. This is memory-efficient
        and fast for dot-product similarity.
        
        Args:
            tokens: List of tokens from the document
            vocab: Term → index mapping (shared across all documents)
            
        Returns:
            SparseVector with indices and values
        """
        term_freq = Counter(tokens)
        
        indices = []
        values = []
        for term, freq in term_freq.items():
            if term in vocab:
                indices.append(vocab[term])
                values.append(float(freq))
        
        return SparseVector(indices=indices, values=values)
    
    def index_repo(
        self, 
        repo_path: str, 
        repo_id: str,
        progress_callback: Optional[callable] = None
    ) -> int:
        """
        Full indexing pipeline for a repository.
        
        Steps:
        1. Walk directory, collect source files
        2. Parse each file with AST chunker
        3. Build BM25 vocabulary over all chunks
        4. Embed each chunk (dense vectors)
        5. Compute BM25 scores (sparse vectors)
        6. Upsert both into Qdrant
        7. Save vocabulary for query-time use
        
        Args:
            repo_path: Local path to the cloned repository
            repo_id: Unique identifier for the repository
            progress_callback: Optional function(status, progress%) for updates
            
        Returns:
            Total number of chunks indexed
        """
        def report(status: str, progress: int = 0):
            if progress_callback:
                progress_callback(status, progress)
            print(f"[{repo_id}] {status} ({progress}%)")
        
        # Step 1: Collect source files
        report("Collecting source files...", 5)
        source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx'}
        source_files = []
        repo = Path(repo_path)
        
        for file_path in repo.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix not in source_extensions:
                continue
            rel_path = str(file_path.relative_to(repo))
            if should_skip_path(rel_path):
                continue
            source_files.append(file_path)
        
        report(f"Found {len(source_files)} source files", 10)
        
        # Step 2: Chunk all files
        report("Parsing ASTs and chunking files...", 15)
        all_chunks: list[Chunk] = []
        
        for file_path in source_files:
            try:
                source_code = file_path.read_text(encoding='utf-8', errors='ignore')
                ext = file_path.suffix
                rel_path = str(file_path.relative_to(repo))
                
                # Use chunk_file_with_fallback for robustness
                file_chunks = chunk_file_with_fallback(rel_path, source_code, ext)
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"  Skipping {file_path}: {e}")
        
        if not all_chunks:
            report("No chunks found — repo may be empty or unsupported", 100)
            return 0
        
        report(f"Created {len(all_chunks)} chunks", 30)
        
        # Step 3: Build BM25 vocabulary
        # Tokenize all chunks and build a global term → index mapping
        report("Building BM25 vocabulary...", 35)
        tokenized_corpus = [
            self.tokenize(c.context_header + " " + c.raw_text)
            for c in all_chunks
        ]
        
        vocab: dict[str, int] = {}
        for tokens in tokenized_corpus:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        report(f"Vocabulary: {len(vocab)} unique terms", 40)
        
        # Steps 4 + 5 + 6: Embed, compute BM25, upsert in batches
        # Batching prevents OOM for large repos
        batch_size = 32
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_idx:batch_idx + batch_size]
            batch_tokens = tokenized_corpus[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1
            
            progress = 40 + int(50 * (batch_idx / len(all_chunks)))
            report(
                f"Embedding & indexing batch {current_batch}/{total_batches}...",
                progress
            )
            
            # Dense embeddings for the batch
            texts = [
                f"search_document: {c.context_header}\n\n{c.raw_text}" 
                for c in batch
            ]
            dense_vectors = self.embedder.embed_batch(texts, show_progress=False)
            
            # Build Qdrant points with both dense and sparse vectors
            points = []
            for j, (chunk, tokens, dense_vec) in enumerate(
                zip(batch, batch_tokens, dense_vectors)
            ):
                sparse_vec = self.compute_bm25_vector(tokens, vocab)
                
                point = PointStruct(
                    # Qdrant needs uint64 IDs
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": sparse_vec,
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "node_type": chunk.node_type,
                        "parent_class": chunk.parent_class,
                        "language": chunk.language,
                        "raw_text": chunk.raw_text,
                        "context_header": chunk.context_header,
                        "repo_id": repo_id,
                        "char_count": chunk.char_count,
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        # Step 7: Store vocabulary for query-time BM25 vector computation
        vocab_path = os.path.join(METADATA_DIR, f"{repo_id}_vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        
        report(f"Indexing complete! {len(all_chunks)} chunks indexed", 100)
        return len(all_chunks)
    
    def get_vocab_path(self, repo_id: str) -> str:
        """Get the path to a repo's stored vocabulary file."""
        return os.path.join(METADATA_DIR, f"{repo_id}_vocab.json")
    
    def collection_exists(self) -> bool:
        """Check if the Qdrant collection already exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False
