# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`tensorrt_llm._torch.pyexecutor.trace_log_utils`.

Covers the helpers added on top of the existing ``log_mem_snapshot`` /
``log_tensor_size`` / ``moe_activation_probe`` set:

- ``log_tensor_size_once`` — dedup wrapper around ``log_tensor_size``.
- ``moe_intermediate_size_probe`` — formula-based intermediate size logger
  for the TRTLLM-Gen MoE op call sites.

All probes are gated by ``TLLM_LOG_MEM_PROFILE=1``. The fixture below
re-creates the canonical ``capture_log`` pattern from
``tests/unittest/utils/test_logger.py``: attach a StringIO handler
directly to the ``tensorrt_llm.logger`` singleton (pytest's ``caplog``
cannot capture it because the singleton's underlying logger has
``propagate=False`` and a non-stdlib severity gate).

Tests do not require a GPU device — they only create small CPU tensors.
"""

import logging
from io import StringIO

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import trace_log_utils
from tensorrt_llm._torch.pyexecutor.trace_log_utils import (
    log_tensor_size_once, moe_intermediate_size_probe)
from tensorrt_llm.logger import Logger


@pytest.fixture(autouse=True)
def _reset_probe_state(monkeypatch):
    """Reset module-level dedup sets and capture TRT-LLM singleton output.

    Attaches a StringIO handler to ``Logger()._logger`` and lowers
    ``_min_severity`` to ``"info"`` so the probes' INFO messages reach the
    handler. Restores prior state after each test.

    Tests that exercise the env-unset branch (no log expected) call
    ``monkeypatch.delenv("TLLM_LOG_MEM_PROFILE", raising=False)`` first;
    everything else inherits ``TLLM_LOG_MEM_PROFILE=1``.
    """
    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    trace_log_utils._tensor_size_seen.clear()
    trace_log_utils._moe_intermediate_seen.clear()

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    singleton = Logger()
    singleton._logger.addHandler(handler)
    old_level = singleton._logger.level
    old_min_sev = singleton._min_severity
    singleton._logger.setLevel(logging.DEBUG)
    singleton._min_severity = "info"
    try:
        yield stream
    finally:
        singleton._logger.removeHandler(handler)
        singleton._logger.setLevel(old_level)
        singleton._min_severity = old_min_sev


def _lines(stream, marker):
    """All log lines from the StringIO containing ``marker``."""
    return [
        line for line in stream.getvalue().splitlines() if marker in line
    ]


def _find_line(stream, marker):
    matches = _lines(stream, marker)
    assert matches, (
        f"Expected log line containing {marker!r}; "
        f"saw: {stream.getvalue()!r}")
    return matches[0]


# ---------------------------------------------------------------------------
# log_tensor_size_once
# ---------------------------------------------------------------------------


def test_env_unset_no_log(_reset_probe_state, monkeypatch):
    """When the env var is not '1', helpers must be silent."""
    monkeypatch.delenv("TLLM_LOG_MEM_PROFILE", raising=False)
    log_tensor_size_once("tag1",
                         torch.zeros(4, 4),
                         dedup_key=("k", ))
    moe_intermediate_size_probe("fp8_block_scale",
                                num_tokens=128,
                                top_k=8,
                                num_experts=64,
                                intermediate_size=2048,
                                hidden_size=7168,
                                local_num_experts=8)
    assert _lines(_reset_probe_state, "[mem-profile/") == []


def test_log_tensor_size_once_dedupe(_reset_probe_state):
    """Same dedup_key fires once across repeated calls."""
    log_tensor_size_once("t", torch.zeros(4, 4), dedup_key=("a", ))
    log_tensor_size_once("t", torch.zeros(4, 4), dedup_key=("a", ))
    log_tensor_size_once("t", torch.zeros(4, 4), dedup_key=("a", ))
    assert len(_lines(_reset_probe_state, "[mem-profile/t]")) == 1


def test_log_tensor_size_once_distinct_keys(_reset_probe_state):
    """Different dedup_keys produce distinct log lines."""
    log_tensor_size_once("t", torch.zeros(4, 4), dedup_key=("a", ))
    log_tensor_size_once("t", torch.zeros(4, 4), dedup_key=("b", ))
    assert len(_lines(_reset_probe_state, "[mem-profile/t]")) == 2


def test_log_tensor_size_once_extras_present(_reset_probe_state):
    """Any **extra kwargs surface in the log line as key=value."""
    log_tensor_size_once(
        "dsv4/attn_output",
        torch.zeros(16, 32, dtype=torch.bfloat16),
        layer_idx=7,
        dedup_key=("dsv4_attn", 7, (16, 32)),
    )
    line = _find_line(_reset_probe_state, "[mem-profile/dsv4/attn_output]")
    assert "shape=(16, 32)" in line
    assert "dtype=torch.bfloat16" in line
    assert "layer_idx=7" in line
    # 16 * 32 * 2 bytes / (1<<20) = 1024 / 1048576 ≈ 0.00 MiB (rounds to 2dp)
    assert "size=0.00MiB" in line


# ---------------------------------------------------------------------------
# moe_intermediate_size_probe — formula correctness per path
# ---------------------------------------------------------------------------

# UT independently recomputes the formula (does not call into trace_log_utils.py
# helpers for the expected values). If the helper diverges from this UT, the
# helper is wrong, not the UT.


def _expected_max_padded_tokens(n, k, e, t):
    nk = n * k
    ncgas_filled = min(e, nk)
    ncgas_rem = max(0, nk - e)
    return (ncgas_filled + ncgas_rem // t) * t


def test_moe_intermediate_size_probe_fp8_cap_128(_reset_probe_state):
    """FP8 path: tile cap is 128 (NOT 256). Regression guard for design M1."""
    # N*K/Le = 16384*8/32 = 4096; next_pow2 = 4096; clamp to [8, 128] = 128.
    n, k, e, i, h, le = 16384, 8, 256, 2048, 7168, 32
    moe_intermediate_size_probe("fp8_block_scale",
                                num_tokens=n,
                                top_k=k,
                                num_experts=e,
                                intermediate_size=i,
                                hidden_size=h,
                                local_num_experts=le)
    line = _find_line(_reset_probe_state,
                      "[mem-profile/moe-trtllm-gen/tensors]")
    assert "path=fp8_block_scale" in line
    assert "assumed_tile=128" in line, line
    assert "assumed_tile_set=[8,16,32,64,128]" in line, line
    # M = (256 + (131072-256)//128) * 128 = (256 + 1022) * 128 = 163584
    expected_m = _expected_max_padded_tokens(n, k, e, 128)
    assert expected_m == 163584
    assert f"M={expected_m}" in line
    # gemm1_output (fp8, hidden dim = 2*I = 4096) = M * 4096 bytes
    expected_g1_bytes = expected_m * 2 * i
    expected_g1_mib = expected_g1_bytes / (1 << 20)
    assert f"gemm1_output={expected_g1_mib:.2f}MiB" in line
    # gemm2_output (bf16) = M * H * 2 bytes
    expected_g2_mib = expected_m * h * 2 / (1 << 20)
    assert f"gemm2_output={expected_g2_mib:.2f}MiB" in line
    # FP8 includes activation_output / activation_output_scale.
    assert "activation_output=" in line
    assert "activation_output_scale=" in line


def test_moe_intermediate_size_probe_mxfp4_cap_256(_reset_probe_state):
    """MXFP4 (isMxFp8=True) path: tile cap is 256."""
    # N*K/Le = 16384*8/32 = 4096; next_pow2 = 4096; clamp to [8, 256] = 256.
    n, k, e, i, h, le = 16384, 8, 256, 2048, 7168, 32
    moe_intermediate_size_probe("mxfp4_block_scale",
                                num_tokens=n,
                                top_k=k,
                                num_experts=e,
                                intermediate_size=i,
                                hidden_size=h,
                                local_num_experts=le)
    line = _find_line(_reset_probe_state,
                      "[mem-profile/moe-trtllm-gen/tensors]")
    assert "path=mxfp4_block_scale" in line
    assert "assumed_tile=256" in line
    assert "assumed_tile_set=[8,16,32,64,128,256]" in line
    # M = (256 + (131072-256)//256) * 256 = (256 + 511) * 256 = 196352
    expected_m = _expected_max_padded_tokens(n, k, e, 256)
    assert expected_m == 196352
    assert f"M={expected_m}" in line
    # MXFP4 gemm1_output = M * I * 1 byte (fp8)
    expected_g1_mib = expected_m * i / (1 << 20)
    assert f"gemm1_output={expected_g1_mib:.2f}MiB" in line
    # gemm1_output_scale = swizzled_sf_size(M, I/32) bytes (1B/elem, uint8)
    sf_cols = max(1, i // 32)
    rows_up = ((expected_m + 127) // 128) * 128
    cols_up = ((sf_cols + 3) // 4) * 4
    expected_sf_mib = rows_up * cols_up / (1 << 20)
    assert f"gemm1_output_scale={expected_sf_mib:.2f}MiB" in line
    # MXFP4 has no activation_output (fused activation).
    assert "activation_output=" not in line


def test_moe_intermediate_size_probe_nvfp4_full_line(_reset_probe_state):
    """NVFP4 path: full per-tensor line, all expected keys."""
    n, k, e, i, h, le = 16384, 8, 256, 2048, 7168, 32
    moe_intermediate_size_probe("nvfp4_block_scale",
                                num_tokens=n,
                                top_k=k,
                                num_experts=e,
                                intermediate_size=i,
                                hidden_size=h,
                                local_num_experts=le)
    line = _find_line(_reset_probe_state,
                      "[mem-profile/moe-trtllm-gen/tensors]")
    for key in [
            "path=nvfp4_block_scale",
            "assumed_tile=",
            "assumed_tile_set=[8,16,32,64,128,256]",
            "M=",
            "G1=",
            "G2=",
            "gemm1_output=",
            "gemm1_output_scale=",
            "gemm2_output=",
            "routing_buffers=",
            "bmm_workspace=unknown",
            "sum_visible=",
    ]:
        assert key in line, (key, line)
    # NVFP4 has no activation_output (fused activation).
    assert "activation_output=" not in line


def test_moe_intermediate_size_probe_nvfp4_bits_4(_reset_probe_state):
    """NVFP4 G1 uses 4 bits (E2m1), not 8.

    Choose a tiny N where the ``_min_tokens`` 128 KiB floor wins, so the
    bit-count choice is exposed:

    - floor_4bits = ceildiv(2^20, I * 4) = ceildiv(1048576, 512*4) = 512
    - floor_8bits = ceildiv(2^20, I * 8) = ceildiv(1048576, 512*8) = 256

    With M = 16 (T=8, n*k=2 < num_experts=4, max_ncgas=ncgas_filled=2,
    M = 2 * 8 = 16), 16 < 512 so G1 = floor = 512 for 4-bit. Under an
    incorrect 8-bit calculation, G1 would be 256.
    """
    n, k, e, i, h, le = 1, 2, 4, 512, 512, 2
    moe_intermediate_size_probe("nvfp4_block_scale",
                                num_tokens=n,
                                top_k=k,
                                num_experts=e,
                                intermediate_size=i,
                                hidden_size=h,
                                local_num_experts=le)
    line = _find_line(_reset_probe_state,
                      "[mem-profile/moe-trtllm-gen/tensors]")
    assert "G1=512" in line, line
    assert "M=16" in line, line


def test_moe_intermediate_size_probe_dedupe(_reset_probe_state):
    """Same args fired twice → exactly one log line."""
    args = dict(num_tokens=128,
                top_k=8,
                num_experts=64,
                intermediate_size=2048,
                hidden_size=7168,
                local_num_experts=8)
    moe_intermediate_size_probe("fp8_block_scale", **args)
    moe_intermediate_size_probe("fp8_block_scale", **args)
    assert len(
        _lines(_reset_probe_state,
               "[mem-profile/moe-trtllm-gen/tensors]")) == 1


def test_moe_intermediate_size_probe_unknown_path_silent(_reset_probe_state):
    """Unknown path silently no-ops (defensive)."""
    moe_intermediate_size_probe("not_a_real_path",
                                num_tokens=128,
                                top_k=8,
                                num_experts=64,
                                intermediate_size=2048,
                                hidden_size=7168,
                                local_num_experts=8)
    assert _lines(_reset_probe_state, "[mem-profile/") == []
