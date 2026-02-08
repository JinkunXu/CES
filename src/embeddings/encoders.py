# src/features/encoders.py
import numpy as np
import numpy as np
from typing import List, Dict, Optional
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import torch

# Sentence-Transformers Text Encoder ======
class STTextEncoder:
    """
    Real Text encoder:
    - using multi-linguistic model by default:
    - output L2 normalized numpy.float64 vectors;
    - with LRU caching to speed up repeated calls.
    """
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: Optional[str] = None,                 # "cuda" / "cpu" / None
        batch_size: int = 32,
        normalize: bool = True,
        instruction_prefix: str = "query: ",          
        cache_size: int = 4096,
    ):
        

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.normalize = normalize
        self.instruction_prefix = instruction_prefix
        self.d_h = int(self.model.get_sentence_embedding_dimension())

        # simple LRU cache
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            val = self._cache.pop(key)
            self._cache[key] = val
            return val
        return None

    def _cache_put(self, key: str, val: np.ndarray):
        self._cache[key] = val
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def encode(self, texts: List[str]) -> np.ndarray:
        keys = [self.instruction_prefix + t for t in texts]
        out: List[np.ndarray] = []
        missing_idx: List[int] = []

        # read cache first
        for i, k in enumerate(keys):
            v = self._cache_get(k)
            if v is None:
                missing_idx.append(i)
                out.append(None)  # 占位
            else:
                out.append(v)

        # encode missing ones
        if missing_idx:
            batch_texts = [keys[i] for i in missing_idx]
            emb = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,  # True -> identity vector
                show_progress_bar=False,
            )
            # float64
            emb = emb.astype(np.float64)
            for j, idx in enumerate(missing_idx):
                out[idx] = emb[j]
                self._cache_put(keys[idx], emb[j])

        return np.vstack(out)

    def __call__(self, q: str) -> np.ndarray:
        
        vec = self.encode([q])[0]
        # ensure normalization
        n = float(np.linalg.norm(vec)) or 1.0
        return (vec / n).astype(np.float64)

class FeatEncoder:           # f_feat：manual
    def __init__(self, d_c: int):
        self.d_c = d_c
    def __call__(self, q: str) -> np.ndarray:
        
        v = np.zeros(self.d_c)
        v[0] = min(len(q) / 512.0, 1.0)    
        v[1] = sum(ch.isdigit() for ch in q) / max(len(q),1)
        v[2] = 1.0 if "```" in q else 0.0
        return v
