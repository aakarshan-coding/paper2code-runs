# Scaled assignment: reusable NeuralZip-style setup for BF16 weights

## Method to implement
Implement a small, lossless BF16 model-weight compressor that isolates the paper's two core ideas: reusable shared exponent codebooks and packed adjacent exponents.

BF16 values have 16 stored bits: 1 sign bit, 8 exponent bits, and 7 mantissa bits. The compressor must preserve sign and mantissa bits verbatim and compress only the exponent stream. Decompression must reconstruct the exact original BF16 bit patterns.

### Baseline: simplified ZipNN
The baseline splits every tensor's exponent stream into fixed-size chunks. For each chunk, during every compression call, it:
1. Builds a histogram of the chunk's single exponent values.
2. Builds Huffman code lengths for that chunk.
3. Encodes exponents one by one using that local table.
4. Stores sign/mantissa bits verbatim and accounts for per-chunk Huffman metadata.

This baseline intentionally repeats statistical analysis and code construction for every compression call.

### NeuralZip-style method
The NeuralZip-style method has a setup phase and a post-setup compression phase.

Setup phase, excluded from the post-setup compression metric:
1. Extract BF16 exponent chunks from a source model.
2. Compute each chunk's normalized 256-bin exponent histogram.
3. Greedily cluster chunks by square-root Jensen-Shannon distance. A chunk joins the nearest existing centroid if the distance is at most `js_threshold`; otherwise it starts a new cluster. Centroids are weighted by chunk lengths.
4. For each cluster, count adjacent exponent pairs inside chunk boundaries and select up to `max_packed_symbols` frequent pairs as packed symbols.
5. Build one reusable Huffman code-length table per cluster over all singleton exponents plus selected packed symbols. Use greedy parsing for token counts: if the next two exponents form a selected pair, emit the packed symbol and advance by two; otherwise emit one singleton and advance by one.

Post-setup compression phase, used for the speed claim:
1. Reuse the frozen chunk-to-cluster assignments, packed vocabularies, and Huffman code lengths.
2. Parse each chunk with the frozen packed vocabulary and compute/store the token stream.
3. Do not rebuild histograms or Huffman tables during NeuralZip compression.
4. Store sign/mantissa bits verbatim and enough metadata to decompress exactly.

The implementation may store token streams in Python objects for decompression, but `compressed_bytes` must be a simulated archive size computed from Huffman payload bits plus specified metadata, not from Python object size.

### Huffman code lengths
Implement deterministic Huffman code lengths from positive integer counts. If there is only one symbol, assign length 1. Otherwise repeatedly merge two lowest-weight nodes; ties must be deterministic, e.g. by a monotonically increasing counter. Return only lengths, not necessarily bitstrings.

### Synthetic BF16 data
No downloads are required. Generate synthetic BF16 weights with `numpy.random.default_rng(seed)`. For each tensor and chunk, choose a latent exponent family from `num_families`. Family centers are around BF16 exponents `120 + 2*f`. Generate exponents mostly as repeated local pairs such as `(center, center)`, `(center, center+1)`, and `(center-1, center)`, with occasional nearby random exponents. With probability `drift`, replace an exponent by a random exponent in `[112, 136]`. Generate sign bits uniformly and mantissas uniformly in `[0, 127]`. Combine bits as `(sign << 15) | (exponent << 7) | mantissa`, view as `torch.bfloat16`, and return a dict of tensors.

This synthetic data is deliberately small but has the two properties the paper exploits: chunks form a few similar exponent-distribution groups, and adjacent exponent pairs recur often.

## Scaled experiment plan
For one seed, `run_scaled_experiment` must:
1. Generate a synthetic BF16 model with defaults `num_tensors=8`, `tensor_size=32768`, `num_families=4`, `drift=0.0`.
2. Build a NeuralZip setup with `chunk_size=1024`, `js_threshold=0.05`, and `max_packed_symbols=64`.
3. Compress the model once with simplified ZipNN and once with NeuralZip.
4. Verify exact bitwise decompression for both compressors.
5. Return metrics including compression ratios, deterministic post-setup work units, work-speedup, and measured wall-clock seconds.

Because wall-clock timing can be noisy in grading, the claim test compares the deterministic work proxy `post_setup_work_speedup = zipnn_work_units / neuralzip_work_units`. The work units must count the repeated baseline statistical work and Huffman construction but not NeuralZip setup. A reasonable implementation is: exponent visits for both methods, plus substantial per-chunk histogram/codebook-construction work for ZipNN, and no Huffman-construction work for NeuralZip post-setup.

Public claim seeds: `0, 1, 2`. Hidden variants use additional seeds and perturb data size, drift, chunk size, JS threshold, and packed vocabulary size.

## Claim
At this reduced scale, after its setup is available, NeuralZip must beat the simplified ZipNN baseline by at least `2.0x` on deterministic post-setup compression work while preserving exact BF16 reconstruction. As a sanity check, NeuralZip's simulated compression ratio should not be more than 5 percentage points worse than ZipNN on the public claim settings.