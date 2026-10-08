import pytest

from neuralzip_small import run_scaled_experiment


@pytest.mark.parametrize("seed", [11, 17, 23])
def test_claim_on_unseen_seeds_with_drift(seed):
    metrics = run_scaled_experiment(
        seed,
        num_tensors=7,
        tensor_size=24576,
        num_families=5,
        drift=0.015,
        chunk_size=768,
        js_threshold=0.05,
        max_packed_symbols=48,
        timing_repeats=1,
    )
    assert metrics["zipnn_lossless"] == 1.0
    assert metrics["neuralzip_lossless"] == 1.0
    assert metrics["post_setup_work_speedup"] >= 2.0
    assert metrics["neuralzip_ratio"] <= metrics["zipnn_ratio"] + 0.07


def test_claim_on_different_slice_and_perturbed_hyperparameters():
    metrics = run_scaled_experiment(
        31,
        num_tensors=5,
        tensor_size=40960,
        num_families=4,
        drift=0.03,
        chunk_size=1024,
        js_threshold=0.075,
        max_packed_symbols=32,
        timing_repeats=1,
    )
    assert metrics["zipnn_lossless"] == 1.0
    assert metrics["neuralzip_lossless"] == 1.0
    assert metrics["post_setup_work_speedup"] >= 2.0
    assert metrics["neuralzip_ratio"] <= metrics["zipnn_ratio"] + 0.08
