# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

from tensorrt_llm.llmapi.llm_args import (EwsjfConfig, SjfConfig,
                                          WaitingQueuePolicy)

from ..executor_request_queue import RequestQueueItem


class WaitingQueue(ABC):
    """Abstract base class for waiting queues."""

    @abstractmethod
    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> RequestQueueItem:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue."""
        pass

    @abstractmethod
    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSWaitingQueue(deque, WaitingQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to FCFS policy."""
        self.extend(requests)

    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> RequestQueueItem:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` iterable.
        """
        self.extendleft(requests)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        filtered_requests = [req for req in self if req.id not in request_ids]
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class SJFWaitingQueue(WaitingQueue):
    """Shortest-Job-First waiting queue with wait-time aging.

    Prioritizes shorter requests while using sigmoid-normalized aging to
    prevent starvation. Uses the same algorithm as vLLM's SJF implementation.

    Score = length_weight * length_score + time_weight * time_score
    where:
        length_score = 1 / (1 + exp(norm_scale * (prompt_len - length_median)))
        time_score   = 1 / (1 + exp(-norm_scale * (wait_time - time_median)))
        norm_scale   = 1 / median  (for each dimension)

    Both scores are sigmoid-normalized to (0, 1). Shorter prompts and longer
    wait times yield higher scores.

    Internally maintains two lists:
    - _prepended: requests returned via prepend_request(s), served first
    - _requests: new requests, lazily sorted by SJF score on peek/pop
    """

    def __init__(self, sjf_config: Optional[SjfConfig] = None):
        # _requests is sorted ascending so the best item is at the end (O(1) pop)
        self._requests: list[RequestQueueItem] = []
        self._prepended: deque[RequestQueueItem] = deque()
        self._sorted = False
        self._config = sjf_config or SjfConfig()
        self._arrival_times: dict[int, float] = {}

    def _get_arrival_time(self, item: RequestQueueItem,
                          now: float) -> float:
        arrival = getattr(item.request, "py_arrival_time",
                          None) if item.request else None
        if arrival is not None:
            return arrival
        return self._arrival_times.get(item.id, now)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

    def _compute_score(self, item: RequestQueueItem, now: float) -> float:
        prompt_len = (
            len(item.request.input_token_ids)
            if item.request and item.request.input_token_ids
            else 0
        )
        wait_time = max(0.0, now - self._get_arrival_time(item, now))

        # Sigmoid normalization (vLLM-style)
        # Length: inverse sigmoid (shorter = higher score)
        length_scale = 1.0 / self._config.length_median
        length_score = self._sigmoid(
            -length_scale * (prompt_len - self._config.length_median))

        # Time: forward sigmoid (longer wait = higher score)
        time_scale = 1.0 / self._config.time_median
        time_score = self._sigmoid(
            time_scale * (wait_time - self._config.time_median))

        return (self._config.length_weight * length_score
                + self._config.time_weight * time_score)

    def _ensure_sorted(self) -> None:
        if not self._sorted and self._requests:
            now = time.time()
            # Sort ascending so highest-score item is at the end (O(1) pop)
            self._requests.sort(
                key=lambda item: self._compute_score(item, now))
            self._sorted = True

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue."""
        if (
            not request.request
            or not hasattr(request.request, "py_arrival_time")
            or request.request.py_arrival_time is None
        ):
            self._arrival_times[request.id] = time.time()
        self._requests.append(request)
        self._sorted = False

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue."""
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        """Pop the highest-priority request."""
        if self._prepended:
            item = self._prepended.popleft()
            self._arrival_times.pop(item.id, None)
            return item
        self._ensure_sorted()
        if not self._requests:
            raise IndexError("pop from an empty queue")
        item = self._requests.pop()  # O(1) from end (highest score)
        self._arrival_times.pop(item.id, None)
        return item

    def peek_request(self) -> RequestQueueItem:
        """Peek at the highest-priority request without removing it."""
        if self._prepended:
            return self._prepended[0]
        self._ensure_sorted()
        if not self._requests:
            raise IndexError("peek from an empty queue")
        return self._requests[-1]  # highest score is at the end

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self._prepended.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend requests to the front of the queue.

        Note: Matches FCFSWaitingQueue semantics — uses extendleft which
        reverses the order. The caller (request_utils.py) passes
        reversed(pending_requests), so the net effect is that
        pending_requests appear at the front in their original order.
        """
        self._prepended.extendleft(requests)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        self._prepended = deque(
            req for req in self._prepended if req.id not in request_ids)
        self._requests = [
            req for req in self._requests if req.id not in request_ids]
        for rid in request_ids:
            self._arrival_times.pop(rid, None)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._prepended) or bool(self._requests)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._prepended) + len(self._requests)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue (prepended first, then sorted requests)."""
        self._ensure_sorted()
        return itertools.chain(self._prepended,
                               reversed(self._requests)).__iter__()


class EWSJFWaitingQueue(WaitingQueue):
    """EWSJF (Exponentially Weighted SJF) multi-queue waiting queue.

    Based on the EWSJF paper (arxiv 2601.21758). Partitions requests into
    multiple length-based sub-queues and uses density-weighted scoring to
    select which sub-queue to pop from. This is the paper's core mechanism
    for creating performance-homogeneous batches and reducing head-of-line
    blocking.

    Architecture:
    - Requests are routed to sub-queues by prompt length at add time
    - Each sub-queue is FIFO internally
    - On pop, the tactical scheduler scores each non-empty sub-queue
      using the front (oldest) request, and pops from the highest-scoring one
    - Short-prompt queues get higher queue priority (qf), so they drain
      first unless longer queues have waited long enough (aging via cs)

    Scoring formula (per sub-queue, evaluated on front request):
        score = qf * (w_base + w_urgency * cs + w_fairness * log(b + 1))
    where:
        qf = queue_priority / (b + 1)
            SJF heuristic. queue_priority = num_queues - bucket_index,
            so shorter-prompt queues get higher values.
        cs = wait_time / max(b, 1)
            Cost-normalized compute score. Divides wait time by estimated
            prefill cost (approx. proportional to prompt length).
        log(b + 1) = fairness term
            Grows without bound, guaranteeing eventual service for all
            requests (Theorem A.1: lim_{t→∞} Score → ∞).

    Default sub-queue boundaries (geometric, powers of 2):
        [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    Creating 10 queues: [0,256), [256,512), ..., [65536,∞)
    """

    _DEFAULT_BOUNDARIES = [256, 512, 1024, 2048, 4096, 8192,
                           16384, 32768, 65536]

    def __init__(self, ewsjf_config: Optional[EwsjfConfig] = None):
        self._config = ewsjf_config or EwsjfConfig()
        self._prepended: deque[RequestQueueItem] = deque()
        self._arrival_times: dict[int, float] = {}

        # Build sorted boundary list and sub-queue buckets
        boundaries = self._config.queue_boundaries
        self._boundaries: list[int] = sorted(
            boundaries if boundaries else self._DEFAULT_BOUNDARIES)
        self._num_buckets: int = len(self._boundaries) + 1
        # Each bucket is a FIFO deque. Bucket 0 = shortest prompts.
        self._buckets: list[deque[RequestQueueItem]] = [
            deque() for _ in range(self._num_buckets)]
        self._total_count: int = 0

    @staticmethod
    def _get_prompt_len(item: RequestQueueItem) -> int:
        if item.request and item.request.input_token_ids:
            return len(item.request.input_token_ids)
        return 0

    def _get_arrival_time(self, item: RequestQueueItem,
                          now: float) -> float:
        arrival = getattr(item.request, "py_arrival_time",
                          None) if item.request else None
        if arrival is not None:
            return arrival
        return self._arrival_times.get(item.id, now)

    def _find_bucket(self, prompt_len: int) -> int:
        """Find bucket index for a prompt length via binary search.

        Bucket 0 = [0, boundaries[0])        (shortest)
        Bucket k = [boundaries[k-1], boundaries[k])
        Bucket n = [boundaries[-1], ∞)          (longest)
        """
        lo, hi = 0, len(self._boundaries)
        while lo < hi:
            mid = (lo + hi) // 2
            if prompt_len < self._boundaries[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _queue_priority(self, bucket_idx: int) -> int:
        """Queue priority: higher = more urgent (shorter prompts).

        Bucket 0 (shortest) gets priority num_buckets.
        Bucket num_buckets-1 (longest) gets priority 1.
        No queue gets priority 0, ensuring score > 0 always.
        """
        return self._num_buckets - bucket_idx

    def _compute_bucket_score(self, bucket_idx: int, now: float) -> float:
        """Compute EWSJF score for a sub-queue using its front request."""
        bucket = self._buckets[bucket_idx]
        if not bucket:
            return float('-inf')

        front = bucket[0]
        b = self._get_prompt_len(front)
        wait_time = max(0.0, now - self._get_arrival_time(front, now))

        # Queue factor: higher priority queues (shorter) get larger qf
        qf = self._queue_priority(bucket_idx) / (b + 1)

        # Compute score: wait time normalized by prefill cost ≈ prompt_len
        cs = wait_time / max(b, 1)

        return qf * (self._config.w_base
                     + self._config.w_urgency * cs
                     + self._config.w_fairness * math.log(b + 1))

    def _select_best_bucket(self, now: float) -> int:
        """Select the sub-queue with the highest EWSJF score."""
        best_idx = -1
        best_score = float('-inf')
        for i in range(self._num_buckets):
            if not self._buckets[i]:
                continue
            score = self._compute_bucket_score(i, now)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def add_request(self, request: RequestQueueItem) -> None:
        """Route request to the appropriate sub-queue by prompt length."""
        if (
            not request.request
            or not hasattr(request.request, "py_arrival_time")
            or request.request.py_arrival_time is None
        ):
            self._arrival_times[request.id] = time.time()
        prompt_len = self._get_prompt_len(request)
        bucket_idx = self._find_bucket(prompt_len)
        self._buckets[bucket_idx].append(request)
        self._total_count += 1

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests, routing each to its sub-queue."""
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        """Tactical scheduling: pop from the highest-scoring sub-queue."""
        if self._prepended:
            item = self._prepended.popleft()
            self._arrival_times.pop(item.id, None)
            return item
        if self._total_count == 0:
            raise IndexError("pop from an empty queue")
        now = time.time()
        best_idx = self._select_best_bucket(now)
        item = self._buckets[best_idx].popleft()
        self._total_count -= 1
        self._arrival_times.pop(item.id, None)
        return item

    def peek_request(self) -> RequestQueueItem:
        """Peek at the front request of the highest-scoring sub-queue."""
        if self._prepended:
            return self._prepended[0]
        if self._total_count == 0:
            raise IndexError("peek from an empty queue")
        now = time.time()
        best_idx = self._select_best_bucket(now)
        return self._buckets[best_idx][0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self._prepended.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend requests to the front of the queue."""
        self._prepended.extendleft(requests)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs from all sub-queues."""
        self._prepended = deque(
            req for req in self._prepended if req.id not in request_ids)
        for i in range(self._num_buckets):
            old_len = len(self._buckets[i])
            self._buckets[i] = deque(
                req for req in self._buckets[i]
                if req.id not in request_ids)
            self._total_count -= (old_len - len(self._buckets[i]))
        for rid in request_ids:
            self._arrival_times.pop(rid, None)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._prepended) or self._total_count > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._prepended) + self._total_count

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over all sub-queues, shortest-prompt bucket first."""
        chains: list[Iterable[RequestQueueItem]] = [self._prepended]
        for bucket in self._buckets:
            if bucket:
                chains.append(bucket)
        return itertools.chain(*chains).__iter__()


def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    priority_fn: Optional[Callable[[RequestQueueItem], float]] = None,
    sjf_config: Optional[SjfConfig] = None,
    ewsjf_config: Optional[EwsjfConfig] = None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use.
        priority_fn: Reserved for future use.
        sjf_config: Configuration for SJF scheduling. Only used when
            policy is SJF.
        ewsjf_config: Configuration for EWSJF scheduling. Only used when
            policy is EWSJF.

    Returns:
        A WaitingQueue instance.
    """
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()
    elif policy == WaitingQueuePolicy.SJF:
        return SJFWaitingQueue(sjf_config)
    elif policy == WaitingQueuePolicy.EWSJF:
        return EWSJFWaitingQueue(ewsjf_config)
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
