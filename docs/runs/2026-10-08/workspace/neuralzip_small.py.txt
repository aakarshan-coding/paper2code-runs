"""Small reproduction of a NeuralZip-style lossless BF16 weight compressor.

Only the 8-bit exponent stream is entropy coded; sign and mantissa bits are stored
verbatim (8 bits per element). The baseline (simplified ZipNN) rebuilds a
histogram and Huffman table for every chunk on every compression call. NeuralZip
fits a reusable setup (clustered chunk histograms, packed adjacent-exponent pairs,
one Huffman table per cluster) and only parses/encodes after setup.
"""
from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

NUM_EXP = 256
ESCAPE = -1  # escape symbol id in NeuralZip codebooks
_GEN_BLOCK = 1024  # latent-family block length used by the synthetic generator


# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
def generate_synthetic_bf16_model(seed: int, num_tensors: int = 8, tensor_size: int = 32768,
                                  num_families: int = 4, drift: float = 0.0) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    model: Dict[str, torch.Tensor] = {}
    for t in range(num_tensors):
        exps = np.empty(tensor_size, dtype=np.int64)
        pos = 0
        while pos < tensor_size:
            blen = min(_GEN_BLOCK, tensor_size - pos)
            f = int(rng.integers(0, num_families))
            c = 120 + 2 * f
            npairs = (blen + 1) // 2
            kinds = rng.choice(4, size=npairs, p=[0.4, 0.3, 0.2, 0.1])
            a = np.full(npairs, c, dtype=np.int64)
            b = np.full(npairs, c, dtype=np.int64)
            b[kinds == 1] = c + 1
            a[kinds == 2] = c - 1
            rnd = kinds == 3
            a[rnd] = c + rng.integers(-3, 4, size=int(rnd.sum()))
            b[rnd] = c + rng.integers(-3, 4, size=int(rnd.sum()))
            block = np.stack([a, b], axis=1).reshape(-1)[:blen]
            exps[pos:pos + blen] = block
            pos += blen
        if drift > 0:
            mask = rng.random(tensor_size) < drift
            exps[mask] = rng.integers(112, 137, size=int(mask.sum()))
        exps = np.clip(exps, 0, 255)
        sign = rng.integers(0, 2, size=tensor_size, dtype=np.int64)
        mant = rng.integers(0, 128, size=tensor_size, dtype=np.int64)
        bits = (sign << 15) | (exps << 7) | mant
        bits = np.where(bits >= 32768, bits - 65536, bits).astype(np.int16)
        model[f"layer{t}.weight"] = torch.from_numpy(bits).view(torch.bfloat16).clone()
    return model


# ----------------------------------------------------------------------------
# BF16 fields
# ----------------------------------------------------------------------------
def _u16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def extract_bf16_fields(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor.dtype != torch.bfloat16:
        raise TypeError("expected bfloat16 tensor")
    u = _u16(tensor.reshape(-1))
    sign = (u >> 15) & 1
    exponent = (u >> 7) & 0xFF
    mantissa = u & 0x7F
    return sign, exponent, mantissa


def reassemble_bf16(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor,
                    shape: Tuple[int, ...]) -> torch.Tensor:
    s = torch.as_tensor(sign).to(torch.int32).reshape(-1)
    e = torch.as_tensor(exponent).to(torch.int32).reshape(-1)
    m = torch.as_tensor(mantissa).to(torch.int32).reshape(-1)
    bits = ((s & 1) << 15) | ((e & 0xFF) << 7) | (m & 0x7F)
    bits = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16)
    return bits.view(torch.bfloat16).reshape(tuple(shape)).clone()


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ps, qs = p.sum(), q.sum()
    if ps > 0:
        p = p / ps
    if qs > 0:
        q = q / qs
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    js = max(js, 0.0)
    return float(math.sqrt(js))


def _huffman_with_ops(counts: Dict[int, int]) -> Tuple[Dict[int, int], int]:
    """Return Huffman code lengths and a count of elementary construction operations."""
    syms = [s for s, c in counts.items() if c > 0]
    if not syms:
        return {}, 0
    if len(syms) == 1:
        return {syms[0]: 1}, 1
    tie = 0
    heap = []
    for s in sorted(syms):
        heap.append((int(counts[s]), tie, [s]))
        tie += 1
    heapq.heapify(heap)
    lengths = {s: 0 for s in syms}
    k = len(syms)
    ops = k  # heapify
    log_k = max(1, int(math.ceil(math.log2(k))))
    while len(heap) > 1:
        w1, _, s1 = heapq.heappop(heap)
        w2, _, s2 = heapq.heappop(heap)
        for s in s1:
            lengths[s] += 1
        for s in s2:
            lengths[s] += 1
        merged = s1 + s2
        heapq.heappush(heap, (w1 + w2, tie, merged))
        tie += 1
        ops += 3 * log_k + len(merged)
    return lengths, ops


def huffman_code_lengths(counts: Dict[int, int]) -> Dict[int, int]:
    return _huffman_with_ops(counts)[0]


# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------
class NeuralZipSetup:
    def __init__(self, chunk_size: int = 1024, js_threshold: float = 0.05, max_packed_symbols: int = 64) -> None:
        self.chunk_size = int(chunk_size)
        self.js_threshold = float(js_threshold)
        self.max_packed_symbols = int(max_packed_symbols)
        self.assignments: Dict[Tuple[str, int], int] = {}
        self.centroids: List[np.ndarray] = []
        self.packed_pairs: List[List[Tuple[int, int]]] = []
        self.pair_to_symbol: List[Dict[int, int]] = []  # key a*256+b -> symbol id
        self.code_lengths: List[Dict[int, int]] = []
        self.setup_seconds: float = 0.0

    # --- helpers
    def _chunks(self, exps: np.ndarray):
        cs = self.chunk_size
        for i, start in enumerate(range(0, len(exps), cs)):
            yield i, exps[start:start + cs]

    def fit(self, model: Dict[str, torch.Tensor]) -> "NeuralZipSetup":
        t0 = time.perf_counter()
        chunk_list = []  # (key, idx, exps)
        for name, tensor in model.items():
            _, e, _ = extract_bf16_fields(tensor)
            exps = e.numpy().astype(np.int64)
            for i, ch in self._chunks(exps):
                chunk_list.append((name, i, ch))

        # 1-3: histograms + greedy JS clustering with length-weighted centroids
        centroid_mass: List[np.ndarray] = []  # unnormalized weighted sums
        centroid_w: List[float] = []
        self.centroids = []
        members: List[List[np.ndarray]] = []
        for name, i, ch in chunk_list:
            hist = np.bincount(ch, minlength=NUM_EXP).astype(np.float64)
            n = float(len(ch))
            p = hist / n
            best, best_d = -1, float("inf")
            for c_idx, cen in enumerate(self.centroids):
                d = jensen_shannon_distance(p, cen)
                if d < best_d:
                    best, best_d = c_idx, d
            if best >= 0 and best_d <= self.js_threshold:
                centroid_mass[best] += p * n
                centroid_w[best] += n
                self.centroids[best] = centroid_mass[best] / centroid_w[best]
                members[best].append(ch)
            else:
                best = len(self.centroids)
                centroid_mass.append(p * n)
                centroid_w.append(n)
                self.centroids.append(p.copy())
                members.append([ch])
            self.assignments[(name, i)] = best

        # 4-5: packed pairs + Huffman table per cluster
        self.packed_pairs, self.pair_to_symbol, self.code_lengths = [], [], []
        for chunks in members:
            pair_counts = np.zeros(NUM_EXP * NUM_EXP, dtype=np.int64)
            for ch in chunks:
                if len(ch) >= 2:
                    keys = ch[:-1] * NUM_EXP + ch[1:]
                    pair_counts += np.bincount(keys, minlength=NUM_EXP * NUM_EXP)
            nz = np.nonzero(pair_counts >= 2)[0]
            order = sorted(nz.tolist(), key=lambda k: (-int(pair_counts[k]), k))
            chosen = order[:self.max_packed_symbols]
            pairs = [(k // NUM_EXP, k % NUM_EXP) for k in chosen]
            p2s = {k: NUM_EXP + j for j, k in enumerate(chosen)}
            # token counts via greedy parsing
            tok_counts = np.zeros(NUM_EXP + len(chosen), dtype=np.int64)
            for ch in chunks:
                toks = _greedy_parse(ch, p2s)
                tok_counts += np.bincount(toks, minlength=len(tok_counts))
            # observed singletons + selected packed symbols + an escape symbol (followed by
            # 8 raw bits) so exponents unseen at setup remain encodable on new data
            counts = {s: int(c) for s, c in enumerate(tok_counts.tolist()) if c > 0 or s >= NUM_EXP}
            counts[ESCAPE] = 1
            self.packed_pairs.append(pairs)
            self.pair_to_symbol.append(p2s)
            self.code_lengths.append(huffman_code_lengths(counts))
        self.setup_seconds = time.perf_counter() - t0
        return self

    def metadata_bytes(self) -> int:
        total = 0
        for pairs, lens in zip(self.packed_pairs, self.code_lengths):
            # header, pair defs (2 bytes), (symbol, length) per singleton, length per packed/escape
            n_single = sum(1 for k in lens if k < NUM_EXP)
            total += 2 + 2 * len(pairs) + 2 * n_single + (len(lens) - n_single)
        return total


class CompressionResult:
    def __init__(self, method: str, compressed_bytes: int, original_bytes: int, compression_seconds: float,
                 work_units: float, archive: Dict[str, object]) -> None:
        self.method = method
        self.compressed_bytes = int(compressed_bytes)
        self.original_bytes = int(original_bytes)
        self.compression_seconds = float(compression_seconds)
        self.work_units = float(work_units)
        self.archive = archive

    def ratio(self) -> float:
        return self.compressed_bytes / self.original_bytes if self.original_bytes else 0.0


def _greedy_parse(exps, p2s: Dict[int, int], pair_lut: np.ndarray = None) -> np.ndarray:
    """Greedy left-to-right parse: emit packed symbol when (e[i], e[i+1]) is selected, else singleton.

    Vectorized but equivalent to the sequential greedy loop: inside a maximal run of
    positions where a selected pair starts, greedy takes pairs at even offsets from the
    run start (a run can only be entered at its start, since the previous position
    cannot start a pair).
    """
    e = np.asarray(exps, dtype=np.int64)
    n = len(e)
    if pair_lut is None:
        pair_lut = _pair_lut(p2s)
    if n < 2 or not p2s:
        return e.copy()
    keys = e[:-1] * NUM_EXP + e[1:]
    sym = pair_lut[keys]
    v = np.zeros(n, dtype=bool)
    v[:-1] = sym >= 0
    idx = np.arange(n)
    starts = v & ~np.concatenate(([False], v[:-1]))
    run_start = np.maximum.accumulate(np.where(starts, idx, -1))
    take = v & (((idx - run_start) % 2) == 0)
    covered = np.zeros(n, dtype=bool)
    covered[1:] = take[:-1]
    keep = ~covered
    toks = e.copy()
    tidx = np.nonzero(take)[0]
    toks[tidx] = sym[tidx]
    return toks[keep]


def _pair_lut(p2s: Dict[int, int]) -> np.ndarray:
    lut = np.full(NUM_EXP * NUM_EXP, -1, dtype=np.int64)
    for k, v in p2s.items():
        lut[k] = v
    return lut


def _greedy_parse_reference(exps: List[int], p2s: Dict[int, int]) -> List[int]:
    toks = []
    n = len(exps)
    i = 0
    while i < n:
        if i + 1 < n:
            s = p2s.get(exps[i] * NUM_EXP + exps[i + 1])
            if s is not None:
                toks.append(s)
                i += 2
                continue
        toks.append(exps[i])
        i += 1
    return toks


def build_neuralzip_setup(model: Dict[str, torch.Tensor], chunk_size: int = 1024, js_threshold: float = 0.05,
                          max_packed_symbols: int = 64) -> NeuralZipSetup:
    return NeuralZipSetup(chunk_size, js_threshold, max_packed_symbols).fit(model)


# ----------------------------------------------------------------------------
# Compressors
# ----------------------------------------------------------------------------
_TENSOR_HEADER_BYTES = 16   # name id / shape / element count
_CHUNK_HEADER_BYTES = 4     # payload bit length


def compress_zipnn(model: Dict[str, torch.Tensor], chunk_size: int = 1024) -> CompressionResult:
    t0 = time.perf_counter()
    original = 0
    payload_bits = 0
    meta_bytes = 0
    work = 0.0
    tensors = {}
    for name, tensor in model.items():
        original += tensor.numel() * 2
        sign, e, mant = extract_bf16_fields(tensor)
        exps = e.numpy().astype(np.int64)
        chunks = []
        for start in range(0, len(exps), chunk_size):
            ch = exps[start:start + chunk_size]
            n = len(ch)
            # 1. histogram of this chunk (one visit per exponent + scan over 256 bins)
            hist = np.bincount(ch, minlength=NUM_EXP)
            work += n + NUM_EXP
            counts = {int(s): int(hist[s]) for s in np.nonzero(hist)[0]}
            # 2. local Huffman table
            lengths, ops = _huffman_with_ops(counts)
            work += ops
            # 3. encode exponent by exponent
            lut = np.zeros(NUM_EXP, dtype=np.int64)
            for s, l in lengths.items():
                lut[s] = l
            payload_bits += int(lut[ch].sum())
            work += n
            # 4. per-chunk metadata: symbol count + (symbol, length) pairs + header
            meta_bytes += 1 + 2 * len(lengths) + _CHUNK_HEADER_BYTES
            chunks.append({"tokens": ch.astype(np.uint8), "lengths": lengths})
        payload_bits += 8 * len(exps)  # sign + mantissa verbatim
        meta_bytes += _TENSOR_HEADER_BYTES
        tensors[name] = {"shape": tuple(tensor.shape), "sign": sign.to(torch.uint8).clone(),
                         "mantissa": mant.to(torch.uint8).clone(), "chunks": chunks}
    secs = time.perf_counter() - t0
    compressed = int(math.ceil(payload_bits / 8)) + meta_bytes
    archive = {"method": "zipnn", "chunk_size": chunk_size, "tensors": tensors}
    return CompressionResult("zipnn", compressed, original, max(secs, 1e-9), work, archive)


def compress_neuralzip(model: Dict[str, torch.Tensor], setup: NeuralZipSetup) -> CompressionResult:
    t0 = time.perf_counter()
    cs = setup.chunk_size
    original = 0
    payload_bits = 0
    meta_bytes = setup.metadata_bytes()
    work = 0.0
    tensors = {}
    lens_luts, pair_luts = [], []
    for c_idx, lens in enumerate(setup.code_lengths):
        size = NUM_EXP + len(setup.packed_pairs[c_idx])
        esc = lens[ESCAPE]
        lut = np.full(size, esc + 8, dtype=np.int64)  # unseen singleton: escape + 8 raw bits
        for sy, l in lens.items():
            if sy != ESCAPE:
                lut[sy] = l
        lens_luts.append(lut)
        pair_luts.append(_pair_lut(setup.pair_to_symbol[c_idx]))
    for name, tensor in model.items():
        original += tensor.numel() * 2
        sign, e, mant = extract_bf16_fields(tensor)
        exps = e.numpy().astype(np.int64)
        chunks = []
        for ci, start in enumerate(range(0, len(exps), cs)):
            ch = exps[start:start + cs]
            cluster = setup.assignments.get((name, ci))
            if cluster is None:
                # Chunk not seen at setup: route to nearest frozen centroid (no table rebuild).
                p = np.bincount(ch, minlength=NUM_EXP) / float(len(ch))
                cluster = int(np.argmin([jensen_shannon_distance(p, c) for c in setup.centroids]))
                work += len(ch) + NUM_EXP * len(setup.centroids)
            toks = _greedy_parse(ch, setup.pair_to_symbol[cluster], pair_luts[cluster])
            work += len(ch)  # one visit per exponent during parse/encode
            payload_bits += int(lens_luts[cluster][toks].sum())
            meta_bytes += 1 + _CHUNK_HEADER_BYTES  # cluster id + payload length
            chunks.append({"cluster": cluster, "tokens": np.asarray(toks, dtype=np.int32)})
        payload_bits += 8 * len(exps)
        meta_bytes += _TENSOR_HEADER_BYTES
        tensors[name] = {"shape": tuple(tensor.shape), "sign": sign.to(torch.uint8).clone(),
                         "mantissa": mant.to(torch.uint8).clone(), "chunks": chunks}
    secs = time.perf_counter() - t0
    compressed = int(math.ceil(payload_bits / 8)) + meta_bytes
    archive = {"method": "neuralzip", "chunk_size": cs, "tensors": tensors,
               "packed_pairs": [list(p) for p in setup.packed_pairs]}
    return CompressionResult("neuralzip", compressed, original, max(secs, 1e-9), work, archive)


def decompress(result: CompressionResult) -> Dict[str, torch.Tensor]:
    archive = result.archive
    method = archive["method"]
    out = {}
    for name, rec in archive["tensors"].items():
        parts = []
        for ch in rec["chunks"]:
            toks = np.asarray(ch["tokens"], dtype=np.int64)
            if method == "zipnn":
                parts.append(toks)
            else:
                pairs = archive["packed_pairs"][ch["cluster"]]
                seq = []
                for t in toks.tolist():
                    if t < NUM_EXP:
                        seq.append(t)
                    else:
                        a, b = pairs[t - NUM_EXP]
                        seq.append(a)
                        seq.append(b)
                parts.append(np.asarray(seq, dtype=np.int64))
        exps = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        out[name] = reassemble_bf16(rec["sign"], torch.from_numpy(exps), rec["mantissa"], rec["shape"])
    return out


def bitwise_equal_model(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
            return False
        if x.dtype == torch.bfloat16:
            if not torch.equal(_u16(x.reshape(-1)), _u16(y.reshape(-1))):
                return False
        elif not torch.equal(x, y):
            return False
    return True


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def run_scaled_experiment(seed: int, num_tensors: int = 8, tensor_size: int = 32768, num_families: int = 4,
                          drift: float = 0.0, chunk_size: int = 1024, js_threshold: float = 0.05,
                          max_packed_symbols: int = 64, timing_repeats: int = 3) -> Dict[str, float]:
    model = generate_synthetic_bf16_model(seed, num_tensors, tensor_size, num_families, drift)
    setup = build_neuralzip_setup(model, chunk_size, js_threshold, max_packed_symbols)
    reps = max(1, int(timing_repeats))
    z_times, n_times = [], []
    zr = nr = None
    for _ in range(reps):
        zr = compress_zipnn(model, chunk_size)
        z_times.append(zr.compression_seconds)
        nr = compress_neuralzip(model, setup)
        n_times.append(nr.compression_seconds)
    z_ok = bitwise_equal_model(model, decompress(zr))
    n_ok = bitwise_equal_model(model, decompress(nr))
    zs, ns = float(min(z_times)), float(min(n_times))
    return {
        "zipnn_lossless": 1.0 if z_ok else 0.0,
        "neuralzip_lossless": 1.0 if n_ok else 0.0,
        "zipnn_ratio": zr.ratio(),
        "neuralzip_ratio": nr.ratio(),
        "zipnn_compressed_bytes": float(zr.compressed_bytes),
        "neuralzip_compressed_bytes": float(nr.compressed_bytes),
        "original_bytes": float(zr.original_bytes),
        "zipnn_work_units": zr.work_units,
        "neuralzip_work_units": nr.work_units,
        "post_setup_work_speedup": zr.work_units / max(nr.work_units, 1e-12),
        "zipnn_compression_seconds": zs,
        "neuralzip_compression_seconds": ns,
        "wallclock_speedup": zs / max(ns, 1e-12),
        "setup_seconds": float(setup.setup_seconds),
        "num_clusters": float(len(setup.centroids)),
    }
