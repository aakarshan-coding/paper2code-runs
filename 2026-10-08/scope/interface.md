# Interface

Module `neuralzip_small` (file `neuralzip_small.py` at the workspace root). The tests import from it.

```python
def generate_synthetic_bf16_model(seed: int, num_tensors: int = 8, tensor_size: int = 32768, num_families: int = 4, drift: float = 0.0) -> dict[str, torch.Tensor]:
    """Generate a deterministic dict of synthetic BF16 weight tensors with clustered, pair-redundant exponents."""

def extract_bf16_fields(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sign, exponent, and mantissa integer tensors extracted from a BF16 tensor's stored bits."""

def reassemble_bf16(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Reconstruct a BF16 tensor with exactly the supplied sign, exponent, and mantissa bits."""

def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the square-root Jensen-Shannon distance between two normalized histograms."""

def huffman_code_lengths(counts: dict[int, int]) -> dict[int, int]:
    """Compute deterministic Huffman code lengths for symbols with positive integer counts."""

def build_neuralzip_setup(model: dict[str, torch.Tensor], chunk_size: int = 1024, js_threshold: float = 0.05, max_packed_symbols: int = 64) -> NeuralZipSetup:
    """Build reusable clustered Huffman tables and packed-exponent vocabularies for a BF16 model."""

def compress_zipnn(model: dict[str, torch.Tensor], chunk_size: int = 1024) -> CompressionResult:
    """Compress a BF16 model with the simplified ZipNN baseline that rebuilds Huffman tables per chunk."""

def compress_neuralzip(model: dict[str, torch.Tensor], setup: NeuralZipSetup) -> CompressionResult:
    """Compress a BF16 model using a frozen NeuralZip setup without rebuilding codebooks."""

def decompress(result: CompressionResult) -> dict[str, torch.Tensor]:
    """Decompress either compressor's archive and return exactly reconstructed BF16 tensors."""

def bitwise_equal_model(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    """Return True iff two BF16 model dicts have identical keys, shapes, dtypes, and stored bits."""

def run_scaled_experiment(seed: int, num_tensors: int = 8, tensor_size: int = 32768, num_families: int = 4, drift: float = 0.0, chunk_size: int = 1024, js_threshold: float = 0.05, max_packed_symbols: int = 64, timing_repeats: int = 3) -> dict[str, float]:
    """Run the scaled one-seed experiment and return the metrics used by the claim tests."""

class NeuralZipSetup:
    """Frozen reusable setup containing chunk size, cluster assignments, packed vocabularies, and Huffman code lengths."""
    def __init__(self, chunk_size: int = 1024, js_threshold: float = 0.05, max_packed_symbols: int = 64) -> None:
        """Create an empty setup configuration."""

    def fit(self, model: dict[str, torch.Tensor]) -> NeuralZipSetup:
        """Fit the setup to a BF16 model and return self."""


class CompressionResult:
    """Container for a compressed archive, simulated size, exact-decompression metadata, timing, and deterministic work units."""
    def __init__(self, method: str, compressed_bytes: int, original_bytes: int, compression_seconds: float, work_units: float, archive: dict[str, object]) -> None:
        """Create a compression result object."""

    def ratio(self) -> float:
        """Return compressed_bytes divided by original_bytes."""


```
