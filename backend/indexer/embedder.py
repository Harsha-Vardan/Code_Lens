"""
Embedding Model Wrapper.

Handles converting code chunks and user queries into dense vector embeddings
using sentence-transformers. These vectors capture semantic meaning — so
"how does authentication work?" and "verify_token function" produce similar
vectors even though they share no keywords.

Key design decisions:
1. Contextual embedding: we embed context_header + raw_text together, not just
   the raw code. This gives the embedding location awareness.
2. Asymmetric embedding: queries get a "search_query:" prefix while documents
   get a "search_document:" prefix. This is how nomic-embed was trained and
   improves retrieval quality.
3. Normalized embeddings: vectors are L2-normalized so cosine similarity
   equals dot product, enabling faster similarity computation.

Interview: "Why normalize embeddings?"
    "Normalization puts all vectors on the unit sphere — their magnitude becomes 1.
    This means cosine similarity equals dot product similarity. Qdrant's HNSW index
    is optimized for dot product, so normalization lets us use the faster metric."
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union


class Embedder:
    """
    Wraps a sentence-transformers embedding model for both document
    and query embedding.
    
    Uses nomic-ai/nomic-embed-text-v1.5 — a strong open-source embedding model
    that handles both natural language questions and source code text well.
    
    Alternative choices:
    - OpenAI text-embedding-3-small: better quality but costs money
    - cohere embed-v3: good but API-dependent
    - all-MiniLM-L6-v2: smaller/faster but lower quality for code
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize the embedding model.
        
        The model is downloaded on first use and cached locally.
        trust_remote_code=True is needed for nomic models as they use
        custom architecture code.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model = SentenceTransformer(
            model_name, 
            trust_remote_code=True
        )
        self.model_name = model_name
        # Get the output dimension from the model itself
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_chunk(self, chunk) -> np.ndarray:
        """
        Embed a single code chunk with its context header.
        
        CRITICAL design decision: we embed the context_header + raw_text
        together, NOT just the raw_text.
        
        Why? The embedding of "def verify_token(token: str):" alone doesn't
        capture WHERE this function lives (which file, which class).
        Prepending the context header means the embedding encodes both
        the code content AND its location.
        
        This is called "contextual embedding" and it significantly
        improves retrieval quality.
        
        Args:
            chunk: A Chunk object with context_header and raw_text attributes
            
        Returns:
            Normalized numpy array of shape (dimension,)
        """
        text_to_embed = f"search_document: {chunk.context_header}\n\n{chunk.raw_text}"
        return self.model.encode(
            text_to_embed, 
            normalize_embeddings=True
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query for search.
        
        Uses the "search_query:" prefix. nomic-embed-text uses prefixes
        to distinguish query vs document embeddings — this is called
        asymmetric embedding and it improves search quality.
        
        The model was trained with this asymmetry: query embeddings are
        optimized to be close to RELEVANT document embeddings, not to
        be close to other queries.
        
        Args:
            query: The user's natural language question
            
        Returns:
            Normalized numpy array of shape (dimension,)
        """
        prefixed_query = f"search_query: {query}"
        return self.model.encode(
            prefixed_query, 
            normalize_embeddings=True
        )
    
    def embed_batch(
        self, 
        texts: list[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts in batches — much faster than one-at-a-time.
        
        Batching enables GPU parallelism. Even on CPU, it's faster because
        of reduced Python overhead and better memory access patterns.
        
        Args:
            texts: List of strings to embed (should already include prefixes)
            batch_size: Number of texts per batch (32 is good for most GPUs)
            show_progress: Whether to show a progress bar
            
        Returns:
            Numpy array of shape (len(texts), dimension)
        """
        return self.model.encode(
            texts, 
            batch_size=batch_size,
            normalize_embeddings=True, 
            show_progress_bar=show_progress
        )
    
    def embed_texts_for_indexing(self, chunks: list) -> np.ndarray:
        """
        Convenience method: embed a list of Chunk objects for indexing.
        
        Prepends the "search_document:" prefix and includes context headers.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Numpy array of shape (len(chunks), dimension)
        """
        texts = [
            f"search_document: {c.context_header}\n\n{c.raw_text}" 
            for c in chunks
        ]
        return self.embed_batch(texts)
