#!/usr/bin/env python3
import numpy as np

from srsran_live_adapter import (
    SrsranTemporalNRXAdapter,
    build_type1_pilot_positional_encoding,
    channel_estimate_to_nrx,
    received_grid_to_nrx,
)
from temporal_nrx_runtime import TemporalInferenceOutput, TemporalNRXRuntime


def test_tensor_shapes_and_feature_order():
    rx = np.zeros((4, 14, 24), np.complex64)
    y = received_grid_to_nrx(rx)
    assert y.shape == (1, 4, 14, 24)

    h = np.zeros((1, 4, 14, 24), np.complex64)
    for r in range(4):
        h[0, r] = (r + 1) + 1j * (10 + r)
    h_nrx = channel_estimate_to_nrx(h)
    assert h_nrx.shape == (1, 24, 14, 8)
    np.testing.assert_allclose(h_nrx[0, 0, 0, :4], [1, 2, 3, 4])
    np.testing.assert_allclose(h_nrx[0, 0, 0, 4:], [10, 11, 12, 13])


def test_type1_pe_tiles_per_prb():
    pe = build_type1_pilot_positional_encoding(
        num_ues=1,
        num_subcarriers=24,
        num_symbols=14,
        dmrs_symbols=[2, 11],
    )
    assert pe.shape == (1, 24, 14, 2)
    np.testing.assert_allclose(pe[:, :12], pe[:, 12:24])
    assert np.all(np.isfinite(pe))


def test_runtime_memory_is_keyed_by_crnti_and_llr_sign_is_converted():
    d_mem = 4
    seen = []

    def fake_inference(**kwargs):
        prev = np.asarray(kwargs["prev_memory"], np.float32)
        valid = np.asarray(kwargs["memory_valid"], bool)
        pe = np.asarray(kwargs["pilot_positional_encoding"], np.float32)
        seen.append((prev.copy(), valid.copy(), pe.shape))
        llr = np.ones((1, 1, 24, 14, 2), np.float32) * 3.0
        return TemporalInferenceOutput(
            receiver_output=llr,
            next_memory=prev + 1.0,
        )

    runtime = TemporalNRXRuntime(fake_inference, d_mem=d_mem, expiry_slots=8)
    adapter = SrsranTemporalNRXAdapter(runtime)
    rx = np.zeros((2, 14, 24), np.complex64)
    h = np.ones((1, 2, 14, 24), np.complex64)

    out1 = adapter.infer(
        rx_grid=rx,
        channel_estimate=h,
        crntis=[0x4601],
        slot_index=100,
        dmrs_symbols=[2, 11],
    )
    assert out1.llr_grid.shape == (1, 14, 24, 2)
    np.testing.assert_allclose(out1.llr_grid, -3.0)
    assert not seen[0][1][0, 0]

    adapter.infer(
        rx_grid=rx,
        channel_estimate=h,
        crntis=[0x4601],
        slot_index=101,
        dmrs_symbols=[2, 11],
    )
    assert seen[1][1][0, 0]
    np.testing.assert_allclose(seen[1][0][0, 0], np.ones(d_mem))


def main():
    test_tensor_shapes_and_feature_order()
    test_type1_pe_tiles_per_prb()
    test_runtime_memory_is_keyed_by_crnti_and_llr_sign_is_converted()
    print("SRSRAN_LIVE_ADAPTER_TEST_PASSED")


if __name__ == "__main__":
    main()
