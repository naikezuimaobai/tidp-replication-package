"""Relevance scoring using all-MiniLM-L6-v2 with caching and batching"""

from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity

# ========== Local model path ==========
LOCAL_MODEL_PATH = os.environ.get(
    "TIDP_SENTENCE_BERT_MODEL", "/model/bd_gfk/TCDDP/models/all-MiniLM-L6-v2"
)


class RelevanceScorer:
    """Relevance scorer with batch encoding, caching, and normalization."""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        if model_path is None:
            model_path = LOCAL_MODEL_PATH
        print(f"Loading model: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        self.current_query_embedding: Optional[np.ndarray] = None
        self.current_query_text: str = ""
        # Cache text embeddings to avoid repeated encoding.
        self.text_embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def set_query(self, query_text: str):
        """Set the query and cache its embedding."""
        if query_text != self.current_query_text:
            self.current_query_text = query_text
            # Normalization makes the dot product equal to cosine similarity.
            self.current_query_embedding = self.model.encode(
                [query_text], 
                normalize_embeddings=True
            )[0]
    
    def compute_relevance(self, element_text: str, element_type: str = "") -> float:
        """Compute one relevance score; retained for compatibility."""
        if self.current_query_embedding is None:
            return 0.5
        
        if not element_text or len(element_text.strip()) == 0:
            return 0.3
        
        if len(element_text) > 1500:
            element_text = element_text[:1500]
        
        # Check the cache.
        if element_text in self.text_embedding_cache:
            self.cache_hits += 1
            elem_embedding = self.text_embedding_cache[element_text]
        else:
            self.cache_misses += 1
            elem_embedding = self.model.encode([element_text], normalize_embeddings=True)[0]
            self.text_embedding_cache[element_text] = elem_embedding
        
        # The dot product equals cosine similarity for normalized vectors.
        similarity = np.dot(self.current_query_embedding, elem_embedding)
        return max(0.0, min(1.0, float(similarity)))
    
    def compute_batch_relevance(self, element_texts: List[str]) -> List[float]:
        """Compute relevance scores in batches."""
        if self.current_query_embedding is None:
            return [0.5] * len(element_texts)
        
        results = [0.0] * len(element_texts)
        need_encode_texts = []
        need_encode_indices = []
        
        # Step 1: check the cache.
        for i, text in enumerate(element_texts):
            if not text or len(text.strip()) == 0:
                results[i] = 0.3
            elif text in self.text_embedding_cache:
                self.cache_hits += 1
                elem_embedding = self.text_embedding_cache[text]
                similarity = np.dot(self.current_query_embedding, elem_embedding)
                results[i] = max(0.0, min(1.0, float(similarity)))
            else:
                need_encode_texts.append(text)
                need_encode_indices.append(i)
        
        # Step 2: batch-encode new texts.
        if need_encode_texts:
            self.cache_misses += len(need_encode_texts)
            
            # Truncate overly long texts.
            for i in range(len(need_encode_texts)):
                if len(need_encode_texts[i]) > 1500:
                    need_encode_texts[i] = need_encode_texts[i][:1500]
            
            # Batch encoding improves throughput.
            embeddings = self.model.encode(
                need_encode_texts, 
                batch_size=64, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Compute similarities and update the cache.
            for idx, text, emb in zip(need_encode_indices, need_encode_texts, embeddings):
                self.text_embedding_cache[text] = emb
                similarity = np.dot(self.current_query_embedding, emb)
                results[idx] = max(0.0, min(1.0, float(similarity)))
        
        return results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': self.cache_hits + self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def clear_cache(self):
        """Clear the cache and release its references."""
        self.text_embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def precompute_for_elements(self, element_texts: List[str]):
        """Precompute embeddings for repeated queries on the same page."""
        unique_texts = list(set(element_texts))
        need_encode = [t for t in unique_texts if t not in self.text_embedding_cache]
        
        if need_encode:
            # Truncate long texts.
            for i in range(len(need_encode)):
                if len(need_encode[i]) > 1500:
                    need_encode[i] = need_encode[i][:1500]
            
            embeddings = self.model.encode(
                need_encode, 
                batch_size=64, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            for text, emb in zip(need_encode, embeddings):
                self.text_embedding_cache[text] = emb
