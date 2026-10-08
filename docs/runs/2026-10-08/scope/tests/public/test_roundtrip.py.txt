from neuralzip_small import (
    bitwise_equal_model,
    build_neuralzip_setup,
    compress_neuralzip,
    compress_zipnn,
    decompress,
    generate_synthetic_bf16_model,
)


def test_both_compressors_are_lossless_and_nontrivial():
    model = generate_synthetic_bf16_model(seed=123, num_tensors=4, tensor_size=4096, num_families=3)
    setup = build_neuralzip_setup(model, chunk_size=512, js_threshold=0.06, max_packed_symbols=32)
    neural = compress_neuralzip(model, setup)
    zipnn = compress_zipnn(model, chunk_size=512)

    assert bitwise_equal_model(model, decompress(neural))
    assert bitwise_equal_model(model, decompress(zipnn))
    assert neural.original_bytes == sum(t.numel() * 2 for t in model.values())
    assert zipnn.original_bytes == neural.original_bytes
    assert 0 < neural.compressed_bytes < neural.original_bytes
    assert 0 < zipnn.compressed_bytes < zipnn.original_bytes
    assert neural.work_units < zipnn.work_units
    assert neural.ratio() <= zipnn.ratio() + 0.08
