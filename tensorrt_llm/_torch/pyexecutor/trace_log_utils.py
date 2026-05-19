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
        f"size={size_bytes / 1024 / 1024:.2f}MiB{extras}")
