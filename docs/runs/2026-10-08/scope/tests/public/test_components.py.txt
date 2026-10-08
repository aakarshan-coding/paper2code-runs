import math
import numpy as np
import torch

from neuralzip_small import (
    extract_bf16_fields,
    huffman_code_lengths,
    jensen_shannon_distance,
    reassemble_bf16,
)


def _bf16_from_u16(bits):
    signed = [b if b < 32768 else b - 65536 for b in bits]
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16)


def _u16_from_bf16(x):
    return (x.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()


def test_bf16_field_roundtrip_known_values():
    bits = [0x3F80, 0xBF80, 0x4000, 0x0001, 0x7F80]
    x = _bf16_from_u16(bits)
    sign, exponent, mantissa = extract_bf16_fields(x)
    assert sign.tolist() == [0, 1, 0, 0, 0]
    assert exponent.tolist() == [127, 127, 128, 0, 255]
    assert mantissa.tolist() == [0, 0, 0, 1, 0]
    y = reassemble_bf16(sign, exponent, mantissa, tuple(x.shape))
    assert _u16_from_bf16(y) == bits


def test_huffman_lengths_are_prefix_cost_reasonable():
    counts = {0: 8, 1: 4, 2: 2, 3: 1}
    lengths = huffman_code_lengths(counts)
    assert set(lengths) == set(counts)
    assert lengths[0] <= lengths[1] <= lengths[2]
    assert max(lengths.values()) <= 4
    huff_cost = sum(counts[s] * lengths[s] for s in counts)
    fixed_two_bit_cost = 2 * sum(counts.values())
    assert huff_cost < fixed_two_bit_cost


def test_jensen_shannon_distance_properties():
    p = np.array([0.5, 0.5, 0.0], dtype=float)
    q = np.array([0.5, 0.5, 0.0], dtype=float)
    r = np.array([0.0, 0.25, 0.75], dtype=float)
    assert jensen_shannon_distance(p, q) == 0.0
    d1 = jensen_shannon_distance(p, r)
    d2 = jensen_shannon_distance(r, p)
    assert d1 > 0.1
    assert math.isclose(d1, d2, rel_tol=1e-12, abs_tol=1e-12)
