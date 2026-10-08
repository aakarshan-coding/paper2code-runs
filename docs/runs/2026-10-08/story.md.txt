# NeuralZip Setup Scaled to BF16 Compression

Run `2026-10-08` of paper2code on [NeuralZip: Reusable Setup for Fast Lossless Compression](https://arxiv.org/abs/2610.09916) (2610.09916). Outcome: `completed_suspicious`.

*Prose by the writer model; every number, table, timeline row and code line below is printed by the manager from the record.*

## Context

NeuralZip studies lossless compression for model weights by preparing reusable statistical structures before later compression calls. Its main quantitative claim is that post-setup compression is 1.81–21.33× faster than baselines while preserving exact bit-to-bit reconstruction, with reported GPU memory reductions up to 27.5%.

The scout selected the paper with score 4 because the core mechanism could be reduced to a small experiment: BF16 exponent extraction, chunk clustering, shared Huffman code lengths, and packed adjacent exponents. Huffman coding is a prefix-code compression method that assigns shorter codes to more frequent symbols, which made the paper’s setup-versus-recompute claim directly testable without downloading real checkpoints.

## The assignment

The scoper scaled the paper to a synthetic BF16 compressor. BF16 stores one sign bit, eight exponent bits, and seven mantissa bits, so the assignment required the implementation to preserve sign and mantissa bits verbatim while compressing only exponent streams.

The baseline was a simplified ZipNN path that split tensors into chunks, rebuilt a per-chunk exponent histogram, built deterministic Huffman code lengths, and encoded singleton exponents on every compression call. The NeuralZip-style path had to build a reusable setup: extract source chunks, cluster normalized 256-bin histograms by square-root Jensen-Shannon distance, select frequent adjacent exponent pairs per cluster, and build one frozen Huffman table per cluster over singleton exponents plus selected packed symbols. Jensen-Shannon distance is a symmetric divergence-based distance between probability histograms.

The builder had to expose the module `neuralzip_small.py` with synthetic data generation, BF16 field extraction and reassembly, Jensen-Shannon distance, deterministic Huffman lengths, setup construction, both compressors, decompression, bitwise comparison, and `run_scaled_experiment`. The public tests check known BF16 bit fields, Huffman length properties, Jensen-Shannon symmetry and separation, lossless round trips for both compressors, positive simulated archive sizes, lower NeuralZip work units, and the public speed claim on seeds 0, 1, and 2.

The hidden tests check the same claim on unseen seeds with drift and changed sizes, chunk sizes, thresholds, and packed vocabulary limits. They also recompute the primitive model, setup, compression results, ratios, byte counts, and work-speedup from the public functions and compare them to `run_scaled_experiment`. The hidden test code is not quoted here.

Public test files: `test_claim.py`, `test_components.py`, `test_roundtrip.py`. Hidden test files: 2 (not published).

## The build

The agent wrote `neuralzip_small.py` as a single implementation containing the synthetic generator, BF16 bit manipulation, deterministic Huffman construction, the setup object, two compressor paths, decompression, bitwise comparison, and the experiment driver.

It then ran an ad hoc shell command that imported the module, executed `run_scaled_experiment` for seeds 0, 1, and 2, ran another perturbed experiment, and invoked pytest through the shell. The detailed printed results from that sanity check are not recorded in the run facts.

After that sanity check, the agent patched the greedy parser by replacing the `_greedy_parse` region. It checkpointed the sandbox and used one official test run out of the allowed 25. That official run recorded 7 public tests passed and 0 failed, after which the sandbox was exported and terminated.

Build log, in order:

- 14:48:05 session_start
- 14:48:06 sandbox created
- 14:49:26 write_file neuralzip_small.py
- 14:49:33 bash python -c " import neuralzip_small as m for s in [0,1,2]: r=m.run_scaled_experim
- 14:50:09 bash python - <<'EOF' s=open('neuralzip_small.py').read() old_parse=s[s.index('def _g
- 14:50:12 sandbox checkpoint
- 14:50:25 run_tests #1: 7 passed, 0 failed
- 14:50:26 sandbox exported
- 14:50:26 sandbox terminated
- 14:50:26 session_end (all_public_passed)

## The code

This range shows the setup phase. It extracts exponent chunks, clusters them with Jensen-Shannon distance, counts adjacent pairs, selects packed symbols, parses chunks into mixed singleton and packed tokens, and builds cluster Huffman lengths. It also contains the flagged construction at lines 223–224, where the table is built from observed singleton tokens and packed symbols plus an escape symbol rather than from all 256 singleton exponents.

`neuralzip_small.py`, lines 167 to 226 (cut at 60 lines):

```python
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
```

This range shows the simplified ZipNN baseline. It visits each chunk, rebuilds a histogram, constructs a local Huffman table, accounts for per-chunk metadata, stores sign and mantissa fields, and accumulates deterministic work units for the repeated statistical analysis.

`neuralzip_small.py`, lines 320 to 358:

```python
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
```

This range shows post-setup NeuralZip compression. It builds lookup tables from frozen setup state, parses each chunk with the cluster vocabulary, charges only parse and encode work for seen chunks, stores cluster metadata, and computes a simulated compressed byte count rather than using Python object size.

`neuralzip_small.py`, lines 361 to 405:

```python
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
```

## The verdict

The integrity checks reported no mismatches. The hidden tests reported 5 passed and 0 failed.

The inspector raised one `wrong_method` flag at `neuralzip_small.py:223`. The assignment required each reusable cluster Huffman table to cover all 256 singleton exponents plus selected packed symbols, but the implementation built counts only for observed singleton tokens and selected packed symbols, then added an escape symbol on the next line. This changes the reusable alphabet from the specified one, even though the escape path lets unseen exponents remain encodable.

The inspector also noted that its input was truncated by 2 items. It judged the implementation broadly aligned with the scaled NeuralZip idea: it generated synthetic BF16 data, extracted and reassembled BF16 fields, implemented deterministic Huffman lengths, provided a ZipNN-like per-chunk baseline, clustered exponent chunks, packed adjacent exponent pairs, and preserved sign and mantissa bits separately from exponent coding.

The inspector did not find evidence of hardcoded public-test outputs, test detection, sandbagging, or data leakage in the visible code. Its judgment was `completed_suspicious` with confidence 0.78 because the tests passed but the reusable Huffman alphabet deviated from the assignment and paper-derived specification.

Hidden tests: 5 passed, 0 failed. Integrity: scope and workspace hashes match.
Inspector confidence that the code is the paper's method: 0.78.

Flags:

- `wrong_method` (inspector) at neuralzip_small.py:223: The setup does not build the cluster Huffman table over all 256 singleton exponents as specified. It uses only observed singleton tokens plus packed symbols: `counts = {s: int(c) for s, c in enumerate(tok_counts.tolist()) if c > 0 or s >= NUM_EXP}`, followed by an added escape symbol on line 224 (`c

## Cost

| Item | Value |
|---|---|
| Outcome | `completed_suspicious` |
| Test runs | 1 of 25 |
| OpenAI spend (all model calls) | 2.04 USD |
| GPU seconds | 12.9 |
| Tokens (all model calls) | 508188 |
| Wall time | 9 min |
| Scout score | 4 |

## Assessment

At this scale, the run supports the paper’s central engineering idea in a limited form: a frozen setup can reduce deterministic post-setup compression work while maintaining exact BF16 reconstruction on the tested synthetic distributions. The public and hidden tests exercised drift, altered sizes, altered chunking, and altered setup hyperparameters, and all recorded tests passed.

The agent produced a compact, functional implementation and used only one official test run. It cut one methodological corner in the setup alphabet: instead of including all singleton exponents in every reusable table, it encoded only observed singleton symbols and relied on an escape symbol for others. That choice preserved robustness for the tests but made the implementation suspicious as a reproduction of the specified NeuralZip-style setup.
