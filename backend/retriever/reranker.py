"""
Cross-Encoder Re-ranking.

This module rescores the top candidates from hybrid search using a
cross-encoder model. This is the "precision" stage of retrieval.

Why cross-encoders improve results:
- The bi-encoder (embedding model) encodes query and document SEPARATELY,
  then compares vectors. Fast but imprecise.
- The cross-encoder takes query and document AS A PAIR, passes them through
  a transformer TOGETHER, and outputs a single relevance score.
- It can model fine-grained interactions between query tokens and document
  tokens — much more accurate, but much slower.

Pipeline position:
    Bi-encoder: embed(query) · embed(doc) = cosine sim  → fast, approximate (top 20)
    Cross-encoder: transformer(query + doc) = relevance  → slow, precise  (top 5)

We only run the cross-encoder on the top 20 results from RRF, not on ALL
chunks — this keeps latency reasonable (~100ms for 20 candidates on CPU).

Interview: "Could you skip the cross-encoder and just use RRF top-5?"
    "Yes, and for latency-critical applications you'd make that tradeoff.
    But in my eval set, adding re-ranking improved answer faithfulness
    by about 15%. The cross-encoder catches cases where a chunk contains
    the query keywords (BM25 picks it up) but is actually discussing a
    different concept. Those false positives pass RRF but fail re-ranking.
    The extra ~100ms latency is worth it for answer quality."
"""

from sentence_transformers import CrossEncoder
from typing import Optional


class Reranker:
    """
    Cross-encoder re-ranker that rescores retrieval candidates.
    
    Uses ms-marco-MiniLM-L-6-v2, trained on Microsoft MARCO passage
    ranking dataset. It's specifically trained to score query-passage
    relevance — perfect for RAG retrieval.
    
    Small enough to run on CPU in ~100ms for 20 candidates.
    """
    
    def __init__(
        self, 
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        max_length: int = 512
    ):
        """
        Initialize the cross-encoder model.
        
        Args:
            model_name: HuggingFace model identifier for the cross-encoder
            max_length: Maximum token length for input pairs. 512 covers
                        most code chunks; longer chunks are truncated.
        """
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.model_name = model_name
    
    def rerank(
        self, 
        query: str, 
        candidates: list, 
        top_k: int = 5
    ) -> list:
        """
        Re-score candidates using the cross-encoder and return the top_k.
        
        Each candidate is paired with the query and scored by the
        cross-encoder. The model outputs a relevance logit (higher = more
        relevant). Candidates are re-sorted by this score.
        
        Args:
            query: The user's question
            candidates: List of (doc_id, {rrf, payload, ...}) tuples from RRF
            top_k: Number of top results to return after re-ranking
            
        Returns:
            List of dicts with payload, rerank_score, rrf_score, and ranks
        """
        if not candidates:
            return []
        
        # Build (query, document) pairs for the cross-encoder
        # We include the context header for location awareness
        pairs = []
        for candidate in candidates:
            doc_id, doc_data = candidate
            payload = doc_data["payload"]
            document_text = (
                f"{payload['context_header']}\n\n"
                f"{payload['raw_text']}"
            )
            pairs.append((query, document_text))
        
        # Score all pairs in a single batch inference call
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates and sort by re-rank score
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        
        # Build the output format
        results = []
        for rank, (candidate, score) in enumerate(scored[:top_k], start=1):
            doc_id, doc_data = candidate
            results.append({
                "payload": doc_data["payload"],
                "rerank_score": float(score),
                "rrf_score": doc_data["rrf"],
                "dense_rank": doc_data.get("dense_rank"),
                "sparse_rank": doc_data.get("sparse_rank"),
                "final_rank": rank,
            })
        
        return results
