import numpy as np
import pytest
import tensorflow as tf

from neural_rx.benchmark.baselines import covariance_from_channels, lmmse_reduction, normalized_covariance
from neural_rx.benchmark.tbler_sweep import SCORING_START, effective_power, summarize


def test_lmmse_reduction_follows_the_upstream_heuristic():
    assert lmmse_reduction(132) == 6
    assert lmmse_reduction(217) == 7


def test_covariance_factors_keep_unit_time_frequency_and_carry_power():
    rng = np.random.default_rng(1)
    samples = (rng.normal(size=(64, 2, 4, 14, 24))+1j*rng.normal(size=(64, 2, 4, 14, 24)))*np.sqrt(1.5/2)
    values = covariance_from_channels(samples.astype(np.complex64))
    result = normalized_covariance(values, 1)
    assert result["time"].shape == (14, 14)
    assert result["frequency"].shape == (24, 24)
    assert result["space"].shape == (4, 4)
    assert np.mean(np.real(np.diag(result["time"]))) == pytest.approx(1)
    assert np.mean(np.real(np.diag(result["frequency"]))) == pytest.approx(1)
    assert np.mean(np.real(np.diag(result["space"]))) == pytest.approx(result["power"])
    assert result["power"] == pytest.approx(1.5, rel=.05)


def test_effective_power_applies_the_precoder_per_tb_and_ue():
    # [batch, TB, rx, rx antenna, tx, tx antenna, symbol, subcarrier]
    channel = np.ones((1, 3, 1, 4, 2, 2, 14, 12), np.complex64)
    channel[0, :, :, :, 1] *= 2
    w = np.ones((2, 2, 1), np.complex64)/np.sqrt(2)
    power = effective_power(tf.constant(channel), tf.constant(w))
    assert power.shape == (3, 2)
    assert power[:, 0] == pytest.approx(2.)
    assert power[:, 1] == pytest.approx(8.)


def _episode(path, errors, fingerprint=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    active = np.ones((64, 2), bool)
    np.savez(path, errors=errors, crc_disagreement=np.zeros_like(errors), active=active,
             ue_ids=np.zeros((64, 2), np.int32), snr_db=np.asarray([0., 1., 2.]), slot_number=np.arange(64),
             channel_power=np.ones((64, 2)), fingerprint=np.full(32, fingerprint, np.uint8))


def test_summarize_scores_only_after_warmup_and_finds_the_target_crossing(tmp_path):
    errors = np.zeros((3, 64, 2), bool)
    errors[:, :SCORING_START] = True  # outside the scored window
    errors[0, SCORING_START:] = True
    _episode(tmp_path/"cond/method/episode_00000.npz", errors)
    summary = summarize(tmp_path)["cond"]["method"]
    assert summary["failures"] == [96, 0, 0]
    assert summary["blocks"] == 96
    assert summary["tbler"] == [1., 0., 0.]
    assert 0 < summary["snr_at_10pct_db"] < 1


def test_summarize_rejects_receivers_decoded_on_different_episodes(tmp_path):
    errors = np.zeros((3, 64, 2), bool)
    _episode(tmp_path/"cond/first/episode_00000.npz", errors, fingerprint=1)
    _episode(tmp_path/"cond/second/episode_00000.npz", errors, fingerprint=2)
    with pytest.raises(ValueError, match="different episodes"):
        summarize(tmp_path)


def test_summarize_merges_sweeps_and_reports_shared_snr_points(tmp_path):
    low = np.zeros((3, 64, 2), bool)
    low[0, SCORING_START:] = True
    _episode(tmp_path/"low/cond/method/episode_00000.npz", low)
    high = np.zeros((2, 64, 2), bool)
    path = tmp_path/"high/cond/method/episode_00000.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, errors=high, crc_disagreement=high, active=np.ones((64, 2), bool),
             snr_db=np.asarray([2., 3.]), fingerprint=np.full(32, 5, np.uint8))
    summary = summarize([tmp_path/"low", tmp_path/"high"])
    assert summary["cond"]["method"]["snr_db"] == [0., 1., 2., 3.]
    assert summary["cond"]["method"]["failures"] == [96, 0, 0, 0]
    assert summary["overlap_points"] == [
        {"condition": "cond", "method": "method", "snr_db": 2., "first": 0, "second": 0}]
