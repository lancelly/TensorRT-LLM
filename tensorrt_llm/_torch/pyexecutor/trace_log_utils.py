"""Gated trace/log utilities for pyexecutor.

Leaf module — no other pyexecutor file is imported here, so any consumer
(``_util``, ``model_engine``, ``model_loader``, ``resource_manager``)
can import freely without creating circular dependencies.
"""

import contextlib
import os

import torch

from tensorrt_llm.logger import logger

_GIB = 1 << 30
_MIB = 1 << 20


def log_mem_snapshot(tag: str) -> None:
    """Log Torch alloc/reserved + alloc/reserved peak + free/total GPU memory.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).

    Prints these fields:

    - ``torch_alloc``         = :func:`torch.cuda.memory_allocated`
    - ``torch_reserved``      = :func:`torch.cuda.memory_reserved`
    - ``torch_alloc_peak``    = :func:`torch.cuda.max_memory_allocated`
    - ``torch_reserved_peak`` = :func:`torch.cuda.max_memory_reserved`
    - ``free``                = ``cuMemGetInfo().free``
    - ``total``               = ``cuMemGetInfo().total``

    Derived quantities the reader may need:

    - ``used      = total - free`` — whole-process GPU consumption
    - ``slack     = reserved - alloc`` — Torch caching allocator free blocks
    - ``non_torch = used - reserved`` — bytes outside Torch (KV pool C++
      cudaMalloc, NCCL buffers, cuBLAS workspace, CUDA driver context,
      CUDA graph mempool, etc.)
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    alloc_peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    logger.info(
        f"[mem-profile/{tag}] "
        f"torch_alloc={alloc / _GIB:.2f}GiB "
        f"torch_reserved={reserved / _GIB:.2f}GiB "
        f"torch_alloc_peak={alloc_peak / _GIB:.2f}GiB "
        f"torch_reserved_peak={reserved_peak / _GIB:.2f}GiB "
        f"free={free / _GIB:.2f}GiB total={total / _GIB:.2f}GiB")


# (path, num_tokens, top_k) tuples already logged — keeps log count bounded.
_moe_probe_seen: set = set()


@contextlib.contextmanager
def moe_activation_probe(path: str, num_tokens: int, top_k: int):
    """Wrap a TRTLLM-Gen MoE op call to log its activation transient peak.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).
    Logs once per unique ``(path, num_tokens, top_k)`` tuple.

    The op allocates gemm1_output / gemm2_output / activation_output etc.
    inside C++ and frees them on return, so the impact is invisible at
    steady state (memory_allocated unchanged across the call) but shows
    up in the **peak** during the call. We measure ``max_memory_allocated``
    before and after; the delta is the additional transient peak this op
    contributed (zero if a prior op already hit a higher peak — that's
    fine, we only care about first-occurrence sizing).
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        yield
        return
    key = (path, int(num_tokens), int(top_k))
    if key in _moe_probe_seen:
        yield
        return
    _moe_probe_seen.add(key)
    peak_before = torch.cuda.max_memory_allocated()
    yield
    peak_after = torch.cuda.max_memory_allocated()
    transient = max(peak_after - peak_before, 0)
    logger.info(
        f"[mem-profile/moe-trtllm-gen/activation] path={path} "
        f"num_tokens={num_tokens} top_k={top_k} "
        f"transient_peak={transient / 1024 / 1024:.2f}MiB")


def log_tensor_size(tag: str, tensor: torch.Tensor, **extra) -> None:
    """Log a single tensor's footprint (shape / dtype / bytes) at a tag.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).

    Bytes = ``numel * element_size``. Any keyword arguments are appended
    as ``key=value`` for caller-specific context (e.g. routing config).
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    size_bytes = tensor.numel() * tensor.element_size()
    extras = "".join(f" {k}={v}" for k, v in extra.items())
    logger.info(
        f"[mem-profile/{tag}] "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"size={size_bytes / _MIB:.2f}MiB{extras}")


# Per-(call-site, key) dedup state for log_tensor_size_once.
_tensor_size_seen: set = set()


def log_tensor_size_once(tag: str,
                         tensor: torch.Tensor,
                         *,
                         dedup_key,
                         **extra) -> None:
    """Dedup wrapper around :func:`log_tensor_size`.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).
    Fires only on the first call for each unique ``dedup_key``; later
    calls with the same key are silent.

    Keeps :func:`log_tensor_size` itself stateless — call sites that want
    a different dedup strategy (or no dedup at all) keep using the bare
    helper.
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    if dedup_key in _tensor_size_seen:
        return
    _tensor_size_seen.add(dedup_key)
    log_tensor_size(tag, tensor, **extra)


# ---- TRTLLM-Gen MoE intermediate size probe -----------------------------------
#
# Computes the expected bytes of the C++-allocated intermediates of a
# TRTLLM-Gen MoE op (gemm1_output, gemm1_output_scale, [activation_output,
# activation_output_scale], gemm2_output, routing buffers) from the Python
# args, using the same formulas as the C++ thop layer
# (cpp/tensorrt_llm/thop/{fp8,fp4,mxFp4}BlockScaleMoe.cpp) and the routing
# helpers in cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h.
#
# This is the static / formula-based companion to ``moe_activation_probe``
# (which measures the dynamic transient_peak via cuda max_memory_allocated).
# Together they let an operator (a) see what the op SHOULD allocate from
# Python-visible args, and (b) compare against what it ACTUALLY allocated.

# Per-path candidate tile sets, matching mSupportedTileN in each C++ runner:
#   - FP8:   fp8BlockScaleMoe.cpp:367           {8, 16, 32, 64, 128}
#   - NVFP4: fp4BlockScaleMoe.cpp:508           {8, 16, 32, 64, 128, 256}
#   - MXFP4: mxFp4BlockScaleMoe.cpp:596         isMxFp8 ? {8,...,256} : {8,...,64}
#     Python only instantiates the MxE4m3MxE2m1 runner with isMxFp8=True
#     (trtllm_gen_custom_ops.py:1208), so we use the {8,...,256} set.
_TILE_SET = {
    "fp8_block_scale": (8, 16, 32, 64, 128),
    "nvfp4_block_scale": (8, 16, 32, 64, 128, 256),
    "mxfp4_block_scale": (8, 16, 32, 64, 128, 256),
}

# Dedup state for moe_intermediate_size_probe. Key tuple matches the args
# that determine the formula output exactly.
_moe_intermediate_seen: set = set()


def _next_pow2(x: int) -> int:
    """Return the smallest power of 2 >= max(1, x)."""
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _ceildiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _round_up(x: int, m: int) -> int:
    return _ceildiv(x, m) * m


def _max_num_padded_tokens(num_tokens: int, top_k: int, num_experts: int,
                           tile_tokens_dim: int) -> int:
    """Mirror of getMaxPermutedPaddedCount + getMaxNumCgasInBatchDim from
    cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h:117-158.
    """
    nk = num_tokens * top_k
    ncgas_filled = min(num_experts, nk)
    ncgas_rem = max(0, nk - num_experts)
    max_ncgas = ncgas_filled + ncgas_rem // tile_tokens_dim
    return max_ncgas * tile_tokens_dim


def _min_tokens(num_padded_tokens: int, hidden_dim: int,
                dtype_bits: int) -> int:
    """Mirror of maybeGetMinTokenCount from runner.h:94-99.

    Pads so total bytes >= 128 KiB.
    """
    if hidden_dim <= 0 or dtype_bits <= 0:
        return num_padded_tokens
    min_required = _ceildiv(128 * 1024 * 8, hidden_dim * dtype_bits)
    return max(num_padded_tokens, min_required)


def _swizzled_sf_size(rows: int, cols: int) -> int:
    """Mirror of computeSwizzledLayoutSFSize from
    cpp/tensorrt_llm/kernels/quantization.h:52-57.
    """
    return _round_up(rows, 128) * _round_up(cols, 4)


def moe_intermediate_size_probe(path: str,
                                num_tokens: int,
                                top_k: int,
                                num_experts: int,
                                intermediate_size: int,
                                hidden_size: int,
                                local_num_experts: int) -> None:
    """Log expected sizes of TRTLLM-Gen MoE C++-allocated intermediates.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).
    Logs once per unique
    ``(path, num_tokens, top_k, num_experts, intermediate_size,
    hidden_size, local_num_experts)`` tuple.

    Assumes the gated-activation branch (SwiGLU-style): ``gemm1_weights``
    second dim is ``2 * intermediate_size``. This is the only variant
    DSv4 (and most current models) exercise.

    ``path`` must be one of: ``"fp8_block_scale"``, ``"nvfp4_block_scale"``,
    ``"mxfp4_block_scale"``. The path-specific tile cap and intermediate
    layout follow the corresponding thop file
    (``cpp/tensorrt_llm/thop/{fp8,fp4,mxFp4}BlockScaleMoe.cpp``).
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    if path not in _TILE_SET:
        return
    key = (path, int(num_tokens), int(top_k), int(num_experts),
           int(intermediate_size), int(hidden_size), int(local_num_experts))
    if key in _moe_intermediate_seen:
        return
    _moe_intermediate_seen.add(key)

    n = int(num_tokens)
    k = int(top_k)
    e = int(num_experts)
    intermediate = int(intermediate_size)
    hidden = int(hidden_size)
    le = max(int(local_num_experts), 1)

    tile_set = _TILE_SET[path]
    # Autotuner fallback (e.g. fp8BlockScaleMoe.cpp:417-418):
    #   T = clamp(next_pow2(num_tokens * top_k / local_num_experts),
    #             mSupportedTileN.front(), mSupportedTileN.back())
    avg = max(1, (n * k) // le)
    t = max(tile_set[0], min(tile_set[-1], _next_pow2(avg)))

    m = _max_num_padded_tokens(n, k, e, t)

    # Per-path G1 / G2 with the right dtype_bits.
    if path == "fp8_block_scale":
        # FP8 path: mDtypeElt = E4m3 (8 bits); gemm1 hidden dim = 2*I.
        g1 = _min_tokens(m, 2 * intermediate, 8)
        g2 = _min_tokens(m, hidden, 16)  # bf16 out
    elif path == "nvfp4_block_scale":
        # NVFP4: mDtypeElt = E2m1 (4 bits); gemm1 hidden dim = I.
        g1 = _min_tokens(m, intermediate, 4)
        g2 = _min_tokens(m, hidden, 16)
    else:  # mxfp4_block_scale (isMxFp8=True): mDtypeAct = MxE4m3 (8 bits)
        g1 = _min_tokens(m, intermediate, 8)
        g2 = _min_tokens(m, hidden, 16)

    # gemm1_output: per-path shape and dtype.
    if path == "fp8_block_scale":
        gemm1_output_bytes = g1 * (2 * intermediate) * 1  # fp8 1 B/elem
        # Both gemm1_output_scale and activation_output{,scale} only on FP8.
        # Layout per fp8BlockScaleMoe.cpp:237-242.
        gemm1_output_scale_bytes = g1 * (2 * intermediate // 128) * 4  # fp32
        activation_output_bytes = g1 * intermediate * 1  # fp8
        activation_output_scale_bytes = g1 * (intermediate // 128) * 4  # fp32
    elif path == "nvfp4_block_scale":
        gemm1_output_bytes = g1 * (intermediate // 2) * 1  # fp8
        sf_cols = max(1, intermediate // 16)
        gemm1_output_scale_bytes = _swizzled_sf_size(g1, sf_cols) * 1  # fp8
        activation_output_bytes = 0
        activation_output_scale_bytes = 0
    else:  # mxfp4_block_scale
        gemm1_output_bytes = g1 * intermediate * 1  # fp8 (Float8_e4m3fn)
        sf_cols = max(1, intermediate // 32)
        gemm1_output_scale_bytes = _swizzled_sf_size(g1, sf_cols) * 1  # uint8
        activation_output_bytes = 0
        activation_output_scale_bytes = 0

    gemm2_output_bytes = g2 * hidden * 2  # bf16

    # Routing buffers — only the dominant ones; small CTA buffers and
    # expert_count_histogram omitted (each is O(KiB)).
    routing_buffers_bytes = (
        n * k * 4              # expanded_idx_to_permuted_idx (int32)
        + m * 4                # permuted_idx_to_token_idx (int32)
        + n * k * 2            # expert_weights (bf16)
        + n * k * 4            # expert_indexes (int32)
    )

    sum_visible_bytes = (gemm1_output_bytes
                        + gemm1_output_scale_bytes
                        + activation_output_bytes
                        + activation_output_scale_bytes
                        + gemm2_output_bytes
                        + routing_buffers_bytes)

    # Build a single-line, grep-able log entry. The shape of the line
    # differs per path (FP8 includes activation_output{,scale}).
    parts = [
        f"path={path}",
        f"N={n}",
        f"K={k}",
        f"E={e}",
        f"I={intermediate}",
        f"H={hidden}",
        f"Le={le}",
        "gated=True",
        f"assumed_tile={t}",
        f"assumed_tile_set=[{','.join(str(x) for x in tile_set)}]",
        f"M={m}",
        f"G1={g1}",
        f"G2={g2}",
        f"gemm1_output={gemm1_output_bytes / _MIB:.2f}MiB",
        f"gemm1_output_scale={gemm1_output_scale_bytes / _MIB:.2f}MiB",
    ]
    if path == "fp8_block_scale":
        parts.append(
            f"activation_output={activation_output_bytes / _MIB:.2f}MiB")
        parts.append(
            f"activation_output_scale={activation_output_scale_bytes / _MIB:.2f}MiB"
        )
    parts.extend([
        f"gemm2_output={gemm2_output_bytes / _MIB:.2f}MiB",
        f"routing_buffers={routing_buffers_bytes / _MIB:.2f}MiB",
        "bmm_workspace=unknown",
        f"sum_visible={sum_visible_bytes / _MIB:.2f}MiB",
    ])
    logger.info("[mem-profile/moe-trtllm-gen/tensors] " + " ".join(parts))
