import pytest

from neuralzip_small import run_scaled_experiment


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_neuralzip_beats_zipnn_post_setup_work(seed):
    metrics = run_scaled_experiment(seed, timing_repeats=2)
    assert metrics["zipnn_lossless"] == 1.0
    assert metrics["neuralzip_lossless"] == 1.0
    assert metrics["post_setup_work_speedup"] >= 2.0
    assert metrics["neuralzip_ratio"] <= metrics["zipnn_ratio"] + 0.05
    assert metrics["zipnn_compressed_bytes"] > 0
    assert metrics["neuralzip_compressed_bytes"] > 0
    assert metrics["zipnn_compression_seconds"] > 0.0
    assert metrics["neuralzip_compression_seconds"] > 0.0
