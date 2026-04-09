import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any

# =========================================================
# 1. Dynamic Base Cache Import
# =========================================================
try:
    from transformers.cache_utils import DynamicCache
    BaseCache = DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    try:
        from transformers.cache_utils import Cache
        BaseCache = Cache
        HAS_DYNAMIC_CACHE = False
    except ImportError:
        class BaseCache(torch.nn.Module):
            def __init__(self):
                super().__init__()
        HAS_DYNAMIC_CACHE = False

# =========================================================
# 2. Robust Local KV Cache
# =========================================================
class LocalKVCache(BaseCache):
    """
    A KV cache implementation that tracks the per-layer sequence dimension.

    Transformers KV layouts can vary across model families and versions.
    Instead of hard-coding assumptions like "seq_len is always at -2" or
    "head_dim is one of {64,128,256}", we infer the concatenation dimension
    from the first observed append for each layer and reuse it consistently.
    """
    def __init__(self):
        try:
            super().__init__()
        except ValueError:
            torch.nn.Module.__init__(self)
        except Exception:
            torch.nn.Module.__init__(self)
            
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Per-layer dimension index along which new tokens are appended.
        # This is inferred lazily on the first non-trivial update for each layer.
        self._seq_dims: List[Optional[int]] = []
        self._seen_tokens = 0
    
    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._seq_dims) <= layer_idx:
            self._seq_dims.append(None)

    def _infer_seq_dim(self, prev_tensor: torch.Tensor, new_tensor: torch.Tensor) -> int:
        """
        Infer which dimension corresponds to sequence length (append dimension).

        Heuristic:
        - Prefer a single dimension that differs between prev and new, with all others equal.
        - If multiple dims differ, prefer the one where new is "small" (often 1..gamma)
          and prev is larger (existing context length).
        - Otherwise fall back to the most common KV layout default for 4D: -2.
        """
        if prev_tensor.ndim != new_tensor.ndim:
            return -2 if prev_tensor.ndim >= 4 else -1

        differing = [d for d in range(prev_tensor.ndim) if prev_tensor.shape[d] != new_tensor.shape[d]]
        if len(differing) == 1:
            return differing[0]

        if differing:
            candidates: List[int] = []
            for d in differing:
                prev_d = int(prev_tensor.shape[d])
                new_d = int(new_tensor.shape[d])
                if new_d <= 16 and prev_d >= new_d:
                    candidates.append(d)
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return candidates[-1]

        if prev_tensor.ndim == 4:
            return -2
        return -1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self._ensure_layer(layer_idx)
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            prev_k = self.key_cache[layer_idx]
            prev_v = self.value_cache[layer_idx]

            seq_dim = self._seq_dims[layer_idx]
            if seq_dim is None:
                seq_dim = self._infer_seq_dim(prev_k, key_states)
                self._seq_dims[layer_idx] = seq_dim

            self.key_cache[layer_idx] = torch.cat([prev_k, key_states], dim=seq_dim)
            self.value_cache[layer_idx] = torch.cat([prev_v, value_states], dim=seq_dim)

        # Best-effort update of seen tokens: use layer 0 length if available.
        try:
            self._seen_tokens = self.get_seq_length(layer_idx=0)
        except Exception:
            pass
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Accurately gets sequence length regardless of dimension order."""
        if len(self.key_cache) <= layer_idx:
            return 0
        
        k = self.key_cache[layer_idx]
        self._ensure_layer(int(layer_idx or 0))
        seq_dim = self._seq_dims[int(layer_idx or 0)]
        if seq_dim is None:
            seq_dim = -2 if k.ndim == 4 else -1
        return int(k.shape[seq_dim])

    def get_usable_length(self, seq_len: int, layer_idx: Optional[int] = 0) -> int:
        return seq_len

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple] = None):
        cache = cls()
        if past_key_values is not None:
            for i, layer_kv in enumerate(past_key_values):
                k, v = layer_kv
                cache.key_cache.append(k)
                cache.value_cache.append(v)
        return cache

# =========================================================
# 3. Model Wrapper
# =========================================================
class KVCacheModel:
    def __init__(self, model : torch.nn.Module) -> None:
        self._model = model
        self._cache = None 
        self._prob_history = None 
        self.device = self._infer_device(model)

    @staticmethod
    def _infer_device(model: torch.nn.Module) -> torch.device:
        # Some HF models don't expose `.device`, and sharded `device_map="auto"`
        # doesn't have a single "model device". This is a best-effort default.
        dev = getattr(model, "device", None)
        if isinstance(dev, torch.device):
            return dev
        for p in model.parameters():
            if isinstance(p, torch.Tensor) and p.device is not None:
                return p.device
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        if self._cache is None:
            outputs = self._model(input_ids, use_cache=use_cache)
            self._prob_history = outputs.logits
            
            if use_cache and outputs.past_key_values is not None:
                if isinstance(outputs.past_key_values, tuple):
                    self._cache = LocalKVCache.from_legacy_cache(outputs.past_key_values)
                else:
                    self._cache = LocalKVCache()
                    if hasattr(outputs.past_key_values, 'key_cache'):
                        self._cache.key_cache = list(outputs.past_key_values.key_cache)
                        self._cache.value_cache = list(outputs.past_key_values.value_cache)
            else:
                self._cache = LocalKVCache()
                
            last_q = self._prob_history[:, -1, :]
        else:
            cached_len = self._cache.get_seq_length()
            if cached_len >= input_ids.shape[1]:
                return self._prob_history[:, -1, :]
            
            current_input_ids = input_ids[:, cached_len:]
            outputs = self._model(
                current_input_ids,
                past_key_values=self._cache if use_cache else None,
                use_cache=use_cache,
            )
            
            new_logits = outputs.logits
            self._prob_history = torch.cat([self._prob_history, new_logits], dim=1)
            last_q = new_logits[:, -1, :]
        
        return last_q

    @torch.no_grad()
    def rollback(self, end_pos: int):
        if self._cache is None:
            return

        # 1. Rollback Logits
        if self._prob_history.shape[1] > end_pos:
            self._prob_history = self._prob_history[:, :end_pos, :]

        # 2. Rollback KV Cache dynamically
        new_key_cache = []
        new_value_cache = []
        
        for layer_idx, (k, v) in enumerate(zip(self._cache.key_cache, self._cache.value_cache)):
            seq_dim = None
            if hasattr(self._cache, "_seq_dims") and len(getattr(self._cache, "_seq_dims")) > layer_idx:
                seq_dim = getattr(self._cache, "_seq_dims")[layer_idx]
            if seq_dim is None:
                seq_dim = -2 if k.ndim == 4 else -1

            if int(k.shape[seq_dim]) >= end_pos:
                k_trim = k.narrow(seq_dim, 0, int(end_pos))
            else:
                k_trim = k

            if int(v.shape[seq_dim]) >= end_pos:
                v_trim = v.narrow(seq_dim, 0, int(end_pos))
            else:
                v_trim = v
                
            new_key_cache.append(k_trim)
            new_value_cache.append(v_trim)
            
        self._cache.key_cache = new_key_cache
        self._cache.value_cache = new_value_cache
        
        if hasattr(self._cache, '_seen_tokens'):
             self._cache._seen_tokens = end_pos

    def get_probs(self):
        return torch.softmax(self._prob_history, dim=-1)