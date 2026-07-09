import unittest

import mlx.core as mx
import numpy as np

from qsdpa_reference import (
    QSDPACase,
    affine_dequantize,
    affine_quantize,
    correctness_cases,
    make_case_inputs,
    normalize_mask,
    quantized_sdpa_dense,
    quantized_sdpa_online,
    routing_boundary_cases,
)


class TestQSDPANumpyReference(unittest.TestCase):
    def test_known_little_endian_packed_layout(self):
        packed4 = np.array([0x76543210] * 4, dtype=np.uint32)
        out4 = affine_dequantize(
            packed4,
            np.ones((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            group_size=32,
            bits=4,
        )
        np.testing.assert_array_equal(out4, np.tile(np.arange(8), 4))

        packed8 = np.array([0x03020100] * 8, dtype=np.uint32)
        out8 = affine_dequantize(
            packed8,
            np.ones((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            group_size=32,
            bits=8,
        )
        np.testing.assert_array_equal(out8, np.tile(np.arange(4), 8))

    def test_affine_round_trip_is_bounded_by_one_scale(self):
        rng = np.random.default_rng(7)
        for bits in (4, 8):
            for group_size in (32, 64):
                values = rng.normal(size=(2, 3, 5, 128)).astype(np.float32)
                packed, scales, biases = affine_quantize(values, group_size, bits)
                restored = affine_dequantize(
                    packed, scales, biases, group_size, bits
                )
                grouped_error = np.abs(values - restored).reshape(
                    *scales.shape, group_size
                )
                self.assertTrue(
                    np.all(grouped_error <= np.abs(scales[..., None]) + 1e-6)
                )

    def test_numpy_dequantizer_matches_mlx_cpu_layout(self):
        rng = np.random.default_rng(8)
        values = rng.normal(size=(1, 2, 5, 128)).astype(np.float32)
        for bits in (4, 8):
            for group_size in (32, 64):
                with self.subTest(bits=bits, group_size=group_size):
                    with mx.stream(mx.cpu):
                        packed, scales, biases = mx.quantize(
                            mx.array(values), group_size=group_size, bits=bits
                        )
                        expected = mx.dequantize(
                            packed,
                            scales,
                            biases,
                            group_size=group_size,
                            bits=bits,
                        )
                        mx.eval(packed, scales, biases, expected)
                    actual = affine_dequantize(
                        np.array(packed),
                        np.array(scales),
                        np.array(biases),
                        group_size,
                        bits,
                    )
                    np.testing.assert_allclose(actual, np.array(expected), atol=1e-6)

    def test_online_softmax_matches_dense_core_matrix(self):
        cases = correctness_cases()
        self.assertEqual(len(cases), 48)
        for case in cases:
            with self.subTest(case=case.name):
                q, qk, qv, mask = make_case_inputs(case)
                kwargs = {
                    "scale": 1.0 / np.sqrt(case.qk_dim),
                    "group_size": case.group_size,
                    "bits": case.bits,
                    "mask": mask,
                }
                expected = quantized_sdpa_dense(q, *qk, *qv, **kwargs)
                actual = quantized_sdpa_online(
                    q, *qk, *qv, block_size=case.block_size, **kwargs
                )
                np.testing.assert_allclose(
                    actual, expected, atol=case.atol, rtol=case.rtol
                )

    def test_reference_supports_distinct_value_dimension(self):
        case = QSDPACase(
            name="value_dim_64",
            seed=9,
            batch=1,
            q_heads=4,
            kv_heads=2,
            q_len=3,
            kv_len=33,
            qk_dim=128,
            value_dim=64,
            bits=4,
            group_size=32,
            mask_kind="causal",
            mask_shape=None,
        )
        q, qk, qv, mask = make_case_inputs(case)
        actual = quantized_sdpa_online(
            q,
            *qk,
            *qv,
            scale=1 / np.sqrt(case.qk_dim),
            group_size=case.group_size,
            bits=case.bits,
            mask=mask,
            block_size=7,
        )
        self.assertEqual(actual.shape, (1, 4, 3, 64))

    def test_batch_two_and_noncontiguous_head_mask(self):
        case = QSDPACase(
            name="batch2_noncontiguous_mask",
            seed=11,
            batch=2,
            q_heads=4,
            kv_heads=2,
            q_len=3,
            kv_len=33,
            qk_dim=128,
            value_dim=128,
            bits=8,
            group_size=64,
            mask_kind="none",
            mask_shape=None,
            block_size=11,
        )
        q, qk, qv, _ = make_case_inputs(case)
        rng = np.random.default_rng(12)
        # Striding a 66-column backing array produces a non-contiguous mask.
        mask_backing = rng.random((2, 4, 3, 66)) > 0.25
        mask = mask_backing[..., ::2]
        self.assertFalse(mask.flags.c_contiguous)
        mask[..., -1] = True
        kwargs = {
            "scale": 1 / np.sqrt(case.qk_dim),
            "group_size": case.group_size,
            "bits": case.bits,
            "mask": mask,
        }
        expected = quantized_sdpa_dense(q, *qk, *qv, **kwargs)
        actual = quantized_sdpa_online(
            q, *qk, *qv, block_size=case.block_size, **kwargs
        )
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_bottom_right_causal_mask(self):
        mask = normalize_mask("causal", batch=1, heads=1, q_len=3, kv_len=5)
        np.testing.assert_array_equal(
            mask,
            np.array(
                [
                    [True, True, True, False, False],
                    [True, True, True, True, False],
                    [True, True, True, True, True],
                ]
            ),
        )

    def test_routing_boundary_metadata_covers_dispatch_risks(self):
        cases = routing_boundary_cases()
        self.assertEqual(len(cases), 198)
        self.assertEqual({case.bits for case in cases}, {4, 8})
        self.assertEqual(
            {case.q_heads // case.kv_heads for case in cases}, {1, 2, 4}
        )
        self.assertEqual(
            {case.q_len for case in cases},
            {1, 4, 8, 16, 31, 32, 33, 96, 127, 128, 129},
        )
        self.assertEqual({case.kv_len for case in cases}, {4096, 16384, 32768})

    def test_validation_rejects_contract_mismatches(self):
        case = correctness_cases()[0]
        q, qk, qv, mask = make_case_inputs(case)
        kwargs = {
            "scale": 1.0,
            "group_size": case.group_size,
            "bits": case.bits,
            "mask": mask,
        }
        q_bad_heads = np.concatenate([q, q[:, :1]], axis=1)
        with self.assertRaisesRegex(ValueError, "divisible"):
            quantized_sdpa_online(q_bad_heads, *qk, *qv, **kwargs)
        with self.assertRaisesRegex(TypeError, "uint32"):
            quantized_sdpa_online(q, qk[0].astype(np.int32), *qk[1:], *qv, **kwargs)
        with self.assertRaisesRegex(ValueError, "broadcastable"):
            quantized_sdpa_online(
                q, *qk, *qv, **{**kwargs, "mask": np.ones((2, 2))}
            )
        with self.assertRaisesRegex(ValueError, "block_size"):
            quantized_sdpa_online(q, *qk, *qv, block_size=0, **kwargs)


if __name__ == "__main__":
    unittest.main()
