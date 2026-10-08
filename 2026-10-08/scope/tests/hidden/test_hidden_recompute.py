import math

from neuralzip_small import (
    bitwise_equal_model,
    build_neuralzip_setup,
    compress_neuralzip,
    compress_zipnn,
    decompress,
    generate_synthetic_bf16_model,
    run_scaled_experiment,
)


def test_experiment_metrics_match_primitive_recomputation():
    params = dict(
        seed=37,
        num_tensors=6,
        tensor_size=28672,
        num_families=4,
        drift=0.02,
        chunk_size=896,
        js_threshold=0.04,
        max_packed_symbols=40,
        timing_repeats=1,
    )
    reported = run_scaled_experiment(**params)

    model = generate_synthetic_bf16_model(
        params["seed"],
        num_tensors=params["num_tensors"],
        tensor_size=params["tensor_size"],
        num_families=params["num_families"],
        drift=params["drift"],
    )
    setup = build_neuralzip_setup(
        model,
        chunk_size=params["chunk_size"],
        js_threshold=params["js_threshold"],
        max_packed_symbols=params["max_packed_symbols"],
    )
    zipnn = compress_zipnn(model, chunk_size=params["chunk_size"])
    neural = compress_neuralzip(model, setup)

    assert bitwise_equal_model(model, decompress(zipnn))
    assert bitwise_equal_model(model, decompress(neural))

    recomputed_zip_ratio = zipnn.compressed_bytes / zipnn.original_bytes
    recomputed_neural_ratio = neural.compressed_bytes / neural.original_bytes
    recomputed_speedup = zipnn.work_units / neural.work_units

    assert reported["zipnn_compressed_bytes"] == zipnn.compressed_bytes
    assert reported["neuralzip_compressed_bytes"] == neural.compressed_bytes
    assert math.isclose(reported["zipnn_ratio"], recomputed_zip_ratio, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(reported["neuralzip_ratio"], recomputed_neural_ratio, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(reported["post_setup_work_speedup"], recomputed_speedup, rel_tol=1e-12, abs_tol=1e-12)
    assert recomputed_speedup >= 2.0
