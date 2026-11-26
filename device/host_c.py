"""The engine's C host side (kernels/libtuff.so: tstore, moe_forward_n, tcache) under the Python card code.

    CHost(w, cfg, store_dir)   cold experts straight from the converted BFP files (tuff_moe: one moe_forward_n call
                               per layer and step, each expert read once per 4 tokens); embeddings and N-gram rows as
                               qfn.Host
    CCache(slots, budget, ...) tcache: the card's expert slots, same policy as qfn.PyCache
    write_store(dir, w, cfg)   what `loader.py convert` writes, for any model size (tests use a tiny one)

    make -C kernels libtuff.so
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import loader  # noqa: E402
import qfn  # noqa: E402

F32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
I32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")


def lib(path=ROOT / "kernels" / "libtuff.so"):
    L = ctypes.CDLL(str(path))
    L.tuff_store_open.restype = ctypes.c_void_p
    L.tuff_store_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.tuff_store_close.argtypes = [ctypes.c_void_p]
    L.tuff_store_experts.argtypes = [ctypes.c_void_p]
    L.tuff_moe.restype = ctypes.c_int
    L.tuff_moe.argtypes = [ctypes.c_void_p, ctypes.c_int, I32, ctypes.c_int, F32, ctypes.c_int, F32, F32]
    L.tuff_cache_new.restype = ctypes.c_void_p
    L.tuff_cache_new.argtypes = [ctypes.c_int] * 4
    L.tuff_cache_free.argtypes = [ctypes.c_void_p]
    L.tuff_cache_lookup.argtypes = [ctypes.c_void_p, I32, ctypes.c_int, I32]
    L.tuff_cache_end.restype = ctypes.c_int
    L.tuff_cache_end.argtypes = [ctypes.c_void_p, I32]
    return L


def write_store(out, w, cfg, fmt="bfp4"):
    """experts.LL.FMT per layer: for each expert, gate_up exp + mant, down exp + mant (loader.convert's layout)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for L in range(cfg.num_hidden_layers):
        p = f"layers.{L}.mlp.experts."
        with open(out / f"experts.{L:02d}.{fmt}", "wb") as fh:
            for e in range(cfg.num_experts):
                for name in ("gate_up_proj", "down_proj"):
                    for part in loader.pack(np.asarray(w[p + name][e], np.float32), fmt):
                        fh.write(part)


class CHost(qfn.Host):
    def __init__(self, w, cfg, store_dir, fmt="bfp4", so=None):
        super().__init__(w, cfg)
        self.L = so or lib()
        self.s = self.L.tuff_store_open(str(store_dir).encode(), loader.MBITS[fmt], cfg.num_hidden_layers,
                                        cfg.hidden_size, cfg.moe_intermediate_size)
        if not self.s:
            raise OSError(f"tuff_store_open({store_dir}) failed")

    def experts(self, L, X, es, G):
        m, d = X.shape
        x = np.ascontiguousarray(X, np.float32)
        y = np.zeros((m, d), np.float32)
        if self.L.tuff_moe(self.s, L, np.asarray(es, np.int32), len(es), np.ascontiguousarray(G, np.float32), m, x, y):
            raise MemoryError("tuff_moe")
        return y.astype(np.float64)

    def __del__(self):
        if getattr(self, "s", None):
            self.L.tuff_store_close(self.s)


class CCache:
    """tcache behind qfn.Card's cache interface: keys (layer, expert) <-> global ids layer * E + expert."""

    def __init__(self, n_slots, budget, n_layers, n_experts, so=None, n_other=0):
        self.L, self.E, self.budget = so or lib(), n_experts, budget
        self.c = self.L.tuff_cache_new(n_layers * n_experts, n_slots, n_other, budget)

    def lookup(self, keys):
        ids = np.array([L * self.E + e for L, e in keys], np.int32)
        slots = np.empty(len(ids), np.int32)
        self.L.tuff_cache_lookup(self.c, ids, len(ids), slots)
        return [int(s) for s in slots]

    def end_step(self):
        out = np.empty(3 * (self.budget + 1), np.int32)
        n = self.L.tuff_cache_end(self.c, out)
        key = lambda i: (int(i) // self.E, int(i) % self.E)
        return [(key(out[3 * i]), None if out[3 * i + 2] < 0 else key(out[3 * i + 2]), int(out[3 * i + 1])) for i in range(n)]

    def __del__(self):
        if getattr(self, "c", None):
            self.L.tuff_cache_free(self.c)
