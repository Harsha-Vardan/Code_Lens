"""
Hybrid Search with Reciprocal Rank Fusion (RRF).

This is the most interview-worthy technical piece. It combines two
fundamentally different search approaches:

1. Dense semantic search: "how does authentication work?" finds semantically
   related code even if it doesn't contain the word "authentication"
   
2. BM25 keyword search: "find the verifyJWT function" does exact term match,
   precise for function names, variable names, specific identifiers

Neither alone is optimal:
- Dense search misses exact names (the embedding for "verifyJWT" might not be
  closest to the exact function)
- BM25 misses semantic queries (searching for "authentication" won't find
  "validate_credentials" even though they're related)

Reciprocal Rank Fusion (RRF) combines rankings from multiple retrieval
systems WITHOUT needing to normalize scores. This is important because
BM25 scores and cosine similarities are on completely different scales —
you can't just add them.

RRF formula:
    RRF_score(doc) = Σ 1/(k + rank_i)
    
    where rank_i is the document's rank in retrieval system i,
    and k is a constant (usually 60) that dampens the impact of top-ranked results.

Interview: "Why k=60?"
    "It's from the original RRF paper (Cormack et al., 2009). The constant prevents
    a document ranked 1st in one list from completely dominating. With k=60, a doc
    ranked 1st gets score 1/61 ≈ 0.016, and 2nd gets 1/62 ≈ 0.016 — the difference
    is tiny. This means being present in BOTH lists matters more than ranking high
    in just one list."
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector, NamedSparseVector,
    SparseVector, Filter, FieldCondition, MatchValue
)
import json
import re
import os
from collections import Counter
from typing import Optional

from backend.indexer.embedder import Embedder
from backend.indexer.pipeline import METADATA_DIR


class HybridSearcher:
    """
    Performs hybrid search combining dense (semantic) and sparse (BM25)
    retrieval with Reciprocal Rank Fusion.
    
    Pipeline:
    1. Embed query → dense vector
    2. Tokenize query → sparse BM25 vector (using stored vocabulary)
    3. Search Qdrant with both vectors independently
    4. Fuse rankings with RRF
    5. Return top candidates for re-ranking
    """
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333", 
        collection_name: str = "codelens_chunks"
    ):
        """
        Args:
            qdrant_url: URL of the Qdrant instance
            collection_name: Name of the Qdrant collection to search
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedder = Embedder()
        self._vocab_cache: dict[str, dict] = {}  # repo_id → vocab
    
    def load_vocab(self, repo_id: str) -> dict:
        """
        Load the BM25 vocabulary for a repository.
        
        Vocabularies are built during indexing and stored as JSON files.
        We cache them in memory to avoid repeated disk reads.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            Dictionary mapping terms to vocabulary indices
        """
        if repo_id not in self._vocab_cache:
            vocab_path = os.path.join(METADATA_DIR, f"{repo_id}_vocab.json")
            with open(vocab_path) as f:
                self._vocab_cache[repo_id] = json.load(f)
        return self._vocab_cache[repo_id]
    
    def tokenize(self, text: str) -> list[str]:
        """Simple tokenizer matching the one used during indexing."""
        return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
    
    def query_to_sparse(self, query: str, vocab: dict) -> SparseVector:
        """
        Convert query text to a sparse BM25 vector using the stored vocabulary.
        
        This MUST use the same vocabulary as the indexed documents. If a query
        term isn't in the vocabulary, it's ignored (it wouldn't match any document
        anyway).
        
        Args:
            query: The user's question text
            vocab: Term → index mapping from indexing
            
        Returns:
            SparseVector with term frequency values
        """
        tokens = self.tokenize(query)
        term_freq = Counter(tokens)
        
        indices = []
        values = []
        for term, freq in term_freq.items():
            if term in vocab:
                indices.append(vocab[term])
                values.append(float(freq))
        
        return SparseVector(indices=indices, values=values)
    
    def reciprocal_rank_fusion(
        self,
        dense_results: list,
        sparse_results: list,
        k: int = 60,
        top_n: int = 20
    ) -> list:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.
        
        For each document that appears in either list, compute:
            RRF_score = Σ 1/(k + rank_in_list_i)
        
        k=60 is the standard constant from the original RRF paper (Cormack et al.).
        It prevents very high-ranked results from completely dominating:
        - A document ranked 1st and 3rd in two lists beats one ranked 1st
          in one list but absent in the other.
        
        Args:
            dense_results: Results from dense (semantic) search
            sparse_results: Results from sparse (BM25) search
            k: RRF smoothing constant (default: 60)
            top_n: Number of fused results to return
            
        Returns:
            List of (doc_id, {"rrf": score, "payload": {...}}) tuples,
            sorted by descending RRF score
        """
        scores: dict = {}
        
        # Score from dense (semantic) ranking
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result.id
            if doc_id not in scores:
                scores[doc_id] = {
                    "rrf": 0.0, 
                    "payload": result.payload,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "dense_score": result.score,
                }
            scores[doc_id]["rrf"] += 1.0 / (k + rank)
        
        # Score from sparse (BM25) ranking
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result.id
            if doc_id not in scores:
                scores[doc_id] = {
                    "rrf": 0.0, 
                    "payload": result.payload,
                    "dense_rank": None,
                    "dense_score": 0.0,
                }
            scores[doc_id]["rrf"] += 1.0 / (k + rank)
            scores[doc_id]["sparse_rank"] = rank
        
        # Sort by combined RRF score (descending)
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf"],
            reverse=True
        )
        
        return sorted_docs[:top_n]
    
    def search(
        self, 
        query: str, 
        repo_id: str, 
        top_k: int = 20,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> list:
        """
        Full hybrid search: dense + sparse + RRF fusion.
        
        Searches the Qdrant collection using both dense (semantic) and
        sparse (BM25) vectors, then fuses the results with RRF.
        
        Results are filtered to only include chunks from the specified repo.
        
        Args:
            query: The user's natural language question
            repo_id: Repository identifier to scope the search
            top_k: Number of candidates to return (for re-ranking)
            dense_weight: Not used in RRF (rank-based), reserved for future
            sparse_weight: Not used in RRF (rank-based), reserved for future
            
        Returns:
            List of (doc_id, {rrf, payload, ...}) tuples, top_k results
        """
        vocab = self.load_vocab(repo_id)
        
        # Filter to only search within the specified repository
        repo_filter = Filter(
            must=[
                FieldCondition(
                    key="repo_id", 
                    match=MatchValue(value=repo_id)
                )
            ]
        )
        
        # Dense search — semantic similarity
        # Embeds the query and finds the nearest vectors in embedding space
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedVector(
                name="dense", 
                vector=query_embedding.tolist()
            ),
            query_filter=repo_filter,
            limit=top_k,
            with_payload=True,
        )
        
        # Sparse search — keyword/BM25 match
        # Converts query to sparse vector and finds matching documents
        query_sparse = self.query_to_sparse(query, vocab)
        sparse_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedSparseVector(
                name="sparse", 
                vector=query_sparse
            ),
            query_filter=repo_filter,
            limit=top_k,
            with_payload=True,
        )
        
        # Fuse with RRF
        fused = self.reciprocal_rank_fusion(
            dense_results, 
            sparse_results,
            top_n=top_k
        )
        
        return fused
