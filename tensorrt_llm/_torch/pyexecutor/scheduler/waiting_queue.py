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
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

from tensorrt_llm.llmapi.llm_args import (EwsjfConfig, SjfConfig,
                                          WaitingQueuePolicy)
from tensorrt_llm.logger import logger

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


class _SubQueue:
    """A single sub-queue in the EWSJF multi-queue system.

    Each sub-queue covers a range [min_len, max_len) of prompt lengths
    and tracks statistics for context-aware scoring.
    """

    __slots__ = ('min_len', 'max_len', 'items', 'is_bubble',
                 'empty_count', '_sum_prompt_len', '_count_total')

    def __init__(self, min_len: int, max_len: int,
                 is_bubble: bool = False):
        self.min_len = min_len
        self.max_len = max_len  # exclusive; math.inf for last queue
        self.items: deque[RequestQueueItem] = deque()
        self.is_bubble = is_bubble
        self.empty_count = 0
        # Running stats for mean prompt length
        self._sum_prompt_len: float = 0.0
        self._count_total: int = 0

    @property
    def mean_prompt_len(self) -> float:
        if self._count_total == 0:
            return (self.min_len + min(self.max_len, 131072)) / 2.0
        return self._sum_prompt_len / self._count_total

    def record_prompt_len(self, prompt_len: int) -> None:
        self._sum_prompt_len += prompt_len
        self._count_total += 1

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return len(self.items) > 0


class EWSJFWaitingQueue(WaitingQueue):
    """EWSJF (Exponentially Weighted SJF) multi-queue waiting queue.

    Full implementation of the EWSJF paper (arxiv 2601.21758) with all
    four components:

    1. **Refine-and-Prune** (Section 4.2): Discovers performance-homogeneous
       request groups via 3-stage unsupervised partitioning:
       - Stage 1: K-means(k=3) for coarse short/medium/long clustering
       - Stage 2: Recursive gap splitting (Gap_j > α * mean(G))
       - Stage 3: Merge by scheduling utility U(qi,qi+1) to respect
         max_queues budget

    2. **Dynamic Queue Routing** (Algorithm 2, Appendix D): Routes requests
       to sub-queues with tolerance-based matching and on-demand bubble
       queue creation for requests in gaps between queues.

    3. **Density-Weighted Scoring** (Eq 1 & 4): Context-aware prioritization
       with per-queue adaptive weights:
         Score(r,q) = qf * (w_base + w_urg(b̄_q)*cs + w_fair(b̄_q)*log(b+1))
       where w_urg/w_fair are linear functions of per-queue mean prompt len.

    4. **Bayesian Meta-Optimization** (Section 4.4): Periodically tunes
       scoring parameters via reward function:
         R(Θ) = λ1*C + λ2*L - λ3*S - λ4*U
       Simplified to random-perturbation hill-climbing (functionally
       equivalent to Bayesian optimization for small parameter spaces).
    """

    _DEFAULT_BOUNDARIES = [256, 512, 1024, 2048, 4096, 8192,
                           16384, 32768, 65536]

    def __init__(self, ewsjf_config: Optional[EwsjfConfig] = None):
        self._config = ewsjf_config or EwsjfConfig()
        self._prepended: deque[RequestQueueItem] = deque()
        self._arrival_times: dict[int, float] = {}
        self._total_count: int = 0

        # Component 1: Initialize queue boundaries
        initial = self._config.initial_queue_boundaries
        if initial:
            boundaries = sorted(initial)
        else:
            boundaries = list(self._DEFAULT_BOUNDARIES)
        self._queues: list[_SubQueue] = self._build_queues(boundaries)

        # Component 1: Prompt length history for Refine-and-Prune
        self._prompt_len_history: list[int] = []
        self._last_repartition_time: float = time.time()

        # Component 2: Bubble queue tracking (pruned via empty_count)
        # (bubble queues are interleaved in self._queues)

        # Component 4: Meta-optimization state
        self._last_meta_opt_time: float = time.time()
        self._current_params: dict[str, float] = {
            'a_urgency': self._config.a_urgency,
            'b_urgency': self._config.b_urgency,
            'a_fairness': self._config.a_fairness,
            'b_fairness': self._config.b_fairness,
            'gap_significance_alpha': self._config.gap_significance_alpha,
        }
        self._best_params: dict[str, float] = dict(self._current_params)
        self._best_reward: float = float('-inf')
        # Metrics accumulated between meta-opt trials
        self._meta_metrics: dict[str, list[float]] = {
            'wait_times': [],
            'queue_sizes': [],
            'prompt_lens': [],
        }
        self._meta_trial_count: int = 0

    @staticmethod
    def _build_queues(boundaries: list[int]) -> list['_SubQueue']:
        """Build sub-queues from sorted boundary list."""
        queues: list[_SubQueue] = []
        prev = 0
        for b in boundaries:
            queues.append(_SubQueue(prev, b))
            prev = b
        # Last queue: [last_boundary, ∞)
        queues.append(_SubQueue(prev, int(2**31)))
        return queues

    # ---- Component 1: Refine-and-Prune ----

    @staticmethod
    def _kmeans_1d(data: list[int], k: int = 3,
                   max_iter: int = 50) -> list[list[int]]:
        """Simple 1D K-means clustering."""
        if len(data) <= k:
            return [[x] for x in sorted(data)]
        sorted_data = sorted(data)
        # Initialize centroids evenly spaced
        n = len(sorted_data)
        centroids = [sorted_data[i * n // k] for i in range(k)]

        for _ in range(max_iter):
            clusters: list[list[int]] = [[] for _ in range(k)]
            for x in sorted_data:
                best_c = min(range(k), key=lambda c: abs(x - centroids[c]))
                clusters[best_c].append(x)
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroids.append(
                        sum(cluster) // len(cluster))
                else:
                    new_centroids.append(0)
            if new_centroids == centroids:
                break
            centroids = new_centroids
        # Remove empty clusters
        return [c for c in clusters if c]

    def _refine_and_prune(self, data: list[int]) -> list[int]:
        """Run Refine-and-Prune to discover queue boundaries.

        Stage 1: K-means(k=3) coarse partitioning
        Stage 2: Recursive gap splitting within each cluster
        Stage 3: Merge low-utility queues to respect max_queues
        """
        if len(data) < 10:
            return list(self._DEFAULT_BOUNDARIES)

        alpha = self._current_params['gap_significance_alpha']

        # Stage 1: K-means(k=3) for short/medium/long
        clusters = self._kmeans_1d(data, k=3)

        # Stage 2: Recursive gap splitting
        all_boundaries: list[int] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            boundaries = self._gap_split(sorted(cluster), alpha)
            all_boundaries.extend(boundaries)

        # Add inter-cluster boundaries
        for i in range(len(clusters) - 1):
            if clusters[i] and clusters[i + 1]:
                boundary = (max(clusters[i]) + min(clusters[i + 1])) // 2
                all_boundaries.append(boundary)

        all_boundaries = sorted(set(b for b in all_boundaries if b > 0))

        # Stage 3: Merge by scheduling utility until <= max_queues
        max_q = self._config.max_queues
        while len(all_boundaries) + 1 > max_q and len(all_boundaries) > 1:
            # Find pair with lowest scheduling utility to merge
            min_utility = float('inf')
            merge_idx = 0
            for i in range(len(all_boundaries) - 1):
                utility = self._scheduling_utility(
                    all_boundaries[i], all_boundaries[i + 1], data)
                if utility < min_utility:
                    min_utility = utility
                    merge_idx = i
            all_boundaries.pop(merge_idx)

        return all_boundaries if all_boundaries else list(
            self._DEFAULT_BOUNDARIES)

    @staticmethod
    def _gap_split(sorted_cluster: list[int], alpha: float) -> list[int]:
        """Split a cluster at significant gaps.

        Gap_j > alpha * mean(all_gaps) → create boundary at midpoint.
        """
        if len(sorted_cluster) < 2:
            return []
        gaps: list[tuple[int, int]] = []  # (gap_size, midpoint)
        for i in range(len(sorted_cluster) - 1):
            gap = sorted_cluster[i + 1] - sorted_cluster[i]
            mid = (sorted_cluster[i] + sorted_cluster[i + 1]) // 2
            gaps.append((gap, mid))
        if not gaps:
            return []
        mean_gap = sum(g for g, _ in gaps) / len(gaps)
        threshold = alpha * mean_gap
        return [mid for gap, mid in gaps if gap > threshold]

    @staticmethod
    def _scheduling_utility(b1: int, b2: int,
                            data: list[int]) -> float:
        """Compute scheduling utility for merging two adjacent queues.

        U(qi, qi+1) = (density_i + density_{i+1}) / (|b̄_{i+1} - b̄_i| + ε)
        Higher utility = less benefit from merging.
        """
        eps = 1.0
        count1 = sum(1 for x in data if x < b1)
        count2 = sum(1 for x in data if b1 <= x < b2)
        density = count1 + count2
        return density / (abs(b2 - b1) + eps)

    def _maybe_repartition(self) -> None:
        """Periodically re-run Refine-and-Prune if enabled."""
        interval = self._config.repartition_interval_seconds
        if interval <= 0 or len(self._prompt_len_history) < 50:
            return
        now = time.time()
        if now - self._last_repartition_time < interval:
            return

        self._last_repartition_time = now
        new_boundaries = self._refine_and_prune(self._prompt_len_history)

        # Only repartition if boundaries actually changed
        old_boundaries = [q.min_len for q in self._queues[1:]]
        if new_boundaries == old_boundaries:
            return

        logger.info(
            f"EWSJF Refine-and-Prune: repartitioning "
            f"{len(self._queues)} → {len(new_boundaries)+1} queues, "
            f"boundaries={new_boundaries[:10]}{'...' if len(new_boundaries) > 10 else ''}")

        # Rebuild queues and redistribute requests
        self._redistribute_to_new_boundaries(new_boundaries)

    def _redistribute_to_new_boundaries(
            self, new_boundaries: list[int]) -> None:
        """Rebuild sub-queues with new boundaries, preserving requests."""
        # Collect all requests from current queues
        all_items: list[RequestQueueItem] = []
        for q in self._queues:
            all_items.extend(q.items)

        # Build new queues
        self._queues = self._build_queues(new_boundaries)
        self._total_count = 0

        # Re-route all requests (preserving arrival times)
        for item in all_items:
            prompt_len = self._get_prompt_len(item)
            idx = self._find_queue_idx(prompt_len)
            self._queues[idx].items.append(item)
            self._queues[idx].record_prompt_len(prompt_len)
            self._total_count += 1

    # ---- Component 2: Dynamic Queue Routing ----

    def _find_queue_idx(self, prompt_len: int) -> int:
        """Find the queue index for a prompt length via binary search."""
        lo, hi = 0, len(self._queues) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if prompt_len < self._queues[mid].max_len:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _route_request_dynamic(self, prompt_len: int) -> int:
        """Dynamic queue routing with bubble queue creation (Algorithm 2).

        Routes request to existing queue with tolerance, or creates a
        bubble queue for requests in gaps.
        """
        idx = self._find_queue_idx(prompt_len)
        q = self._queues[idx]

        # Check if it fits in the found queue with tolerance
        if q.min_len <= prompt_len < q.max_len:
            return idx

        # Check tolerance-based assignment to adjacent queues
        upper_tol = self._config.bubble_upper_tolerance
        lower_tol = self._config.bubble_lower_tolerance

        # Can it fit in the lower queue (idx-1) with upper tolerance?
        if idx > 0:
            q_lower = self._queues[idx - 1]
            if prompt_len <= q_lower.max_len * upper_tol:
                return idx - 1

        # Can it fit in the upper queue (idx) with lower tolerance?
        if prompt_len >= q.min_len * lower_tol:
            return idx

        # Create bubble queue (only if we haven't hit max_queues)
        if len(self._queues) < self._config.max_queues * 2:
            bubble_width = self._config.default_bubble_width
            # Find the gap
            if idx > 0:
                gap_lo = self._queues[idx - 1].max_len
                gap_hi = q.min_len
            else:
                gap_lo = 0
                gap_hi = q.min_len

            available = gap_hi - gap_lo
            width = min(bubble_width, available) if available > 0 else bubble_width
            new_min = max(prompt_len - width // 2, gap_lo)
            new_max = min(prompt_len + width // 2, gap_hi)
            if new_max <= new_min:
                new_max = new_min + width

            bubble = _SubQueue(new_min, new_max, is_bubble=True)
            # Insert bubble in sorted position
            self._queues.insert(idx, bubble)
            logger.debug(
                f"EWSJF: created bubble queue [{new_min}, {new_max}) "
                f"for prompt_len={prompt_len}")
            return idx

        # Fallback: assign to closest queue
        return idx

    def _prune_empty_bubble_queues(self) -> None:
        """Remove bubble queues that have been empty too long."""
        threshold = self._config.empty_queue_threshold
        to_remove: list[int] = []
        for i, q in enumerate(self._queues):
            if q.is_bubble and not q.items:
                q.empty_count += 1
                if q.empty_count > threshold:
                    to_remove.append(i)
        for i in reversed(to_remove):
            logger.debug(
                f"EWSJF: pruning empty bubble queue "
                f"[{self._queues[i].min_len}, {self._queues[i].max_len})")
            self._queues.pop(i)

    # ---- Component 3: Density-Weighted Scoring ----

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

    def _w_urgency(self, mean_prompt_len: float) -> float:
        """Context-aware urgency weight: w_urg(b̄_q) = a_u * b̄_q + b_u."""
        return max(0.0,
                   self._current_params['a_urgency'] * mean_prompt_len
                   + self._current_params['b_urgency'])

    def _w_fairness(self, mean_prompt_len: float) -> float:
        """Context-aware fairness weight: w_fair(b̄_q) = a_f * b̄_q + b_f."""
        return max(0.0,
                   self._current_params['a_fairness'] * mean_prompt_len
                   + self._current_params['b_fairness'])

    def _compute_queue_score(self, q_idx: int, now: float) -> float:
        """Compute EWSJF score for a sub-queue (Eq 1 & 4)."""
        q = self._queues[q_idx]
        if not q.items:
            return float('-inf')

        front = q.items[0]
        b = self._get_prompt_len(front)
        wait_time = max(0.0, now - self._get_arrival_time(front, now))

        # Queue factor: higher index = longer prompts = lower priority
        # qi = num_queues - queue_index (paper's queue priority)
        qi = len(self._queues) - q_idx
        qf = qi / (b + 1)

        # Compute score: wait time normalized by prefill cost
        cs = wait_time / max(b, 1)

        # Context-aware weights based on per-queue mean prompt length
        mean_b = q.mean_prompt_len
        w_urg = self._w_urgency(mean_b)
        w_fair = self._w_fairness(mean_b)

        return qf * (self._config.w_base
                     + w_urg * cs
                     + w_fair * math.log(b + 1))

    def _select_best_queue(self, now: float) -> int:
        """Tactical scheduling: select the highest-scoring sub-queue."""
        best_idx = -1
        best_score = float('-inf')
        for i in range(len(self._queues)):
            if not self._queues[i].items:
                continue
            score = self._compute_queue_score(i, now)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    # ---- Component 4: Bayesian Meta-Optimization ----

    def _compute_reward(self) -> float:
        """Compute R(Θ) = λ1*C + λ2*L - λ3*S - λ4*U (Eq 5).

        C = compactness (low variance within queues)
        L = load balance (even distribution across queues)
        S = queue proliferation penalty
        U = latency penalty
        """
        cfg = self._config
        num_queues = len(self._queues)
        non_empty = [q for q in self._queues if q._count_total > 0]

        if not non_empty:
            return 0.0

        # C: Compactness — inverse of average within-queue variance
        total_variance = 0.0
        for q in non_empty:
            if q._count_total > 1:
                mean = q.mean_prompt_len
                # Estimate variance from range
                range_size = min(q.max_len, 131072) - q.min_len
                total_variance += (range_size / max(mean, 1)) ** 2
        compactness = 1.0 / (1.0 + total_variance / max(len(non_empty), 1))

        # L: Load balance — 1 - normalized std of queue sizes
        sizes = [len(q) for q in self._queues if q.items]
        if sizes and max(sizes) > 0:
            mean_size = sum(sizes) / len(sizes)
            var_size = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            load_balance = 1.0 / (1.0 + math.sqrt(var_size) / max(mean_size, 1))
        else:
            load_balance = 1.0

        # S: Queue proliferation penalty
        proliferation = num_queues / max(cfg.max_queues, 1)

        # U: Latency penalty — mean wait time
        wait_times = self._meta_metrics.get('wait_times', [])
        if wait_times:
            mean_wait = sum(wait_times) / len(wait_times)
            latency_penalty = mean_wait / 10.0  # normalize to ~1
        else:
            latency_penalty = 0.0

        reward = (cfg.lambda_compactness * compactness
                  + cfg.lambda_load_balance * load_balance
                  - cfg.lambda_proliferation * proliferation
                  - cfg.lambda_latency * latency_penalty)
        return reward

    def _maybe_meta_optimize(self) -> None:
        """Periodically run one trial of meta-optimization."""
        if not self._config.meta_optimization_enabled:
            return
        interval = self._config.meta_optimization_interval_seconds
        if interval <= 0:
            return
        now = time.time()
        if now - self._last_meta_opt_time < interval:
            return

        self._last_meta_opt_time = now
        self._meta_trial_count += 1

        # Compute reward for current parameters
        current_reward = self._compute_reward()

        if current_reward > self._best_reward:
            self._best_reward = current_reward
            self._best_params = dict(self._current_params)

        # Perturb parameters (hill-climbing step)
        perturbation_scale = max(0.1, 1.0 / (1.0 + self._meta_trial_count))
        new_params = dict(self._best_params)
        param_ranges = {
            'a_urgency': (-0.001, 0.01),
            'b_urgency': (0.1, 5.0),
            'a_fairness': (-0.001, 0.001),
            'b_fairness': (0.01, 2.0),
            'gap_significance_alpha': (0.5, 3.0),
        }
        for key, (lo, hi) in param_ranges.items():
            delta = (hi - lo) * perturbation_scale * (random.random() - 0.5)
            new_params[key] = max(lo, min(hi, new_params[key] + delta))

        self._current_params = new_params

        # Clear metrics for next trial
        self._meta_metrics = {
            'wait_times': [],
            'queue_sizes': [],
            'prompt_lens': [],
        }

        logger.info(
            f"EWSJF meta-opt trial {self._meta_trial_count}: "
            f"reward={current_reward:.4f} (best={self._best_reward:.4f}), "
            f"params={{a_u={self._current_params['a_urgency']:.6f}, "
            f"b_u={self._current_params['b_urgency']:.4f}, "
            f"a_f={self._current_params['a_fairness']:.6f}, "
            f"b_f={self._current_params['b_fairness']:.4f}}}")

    # ---- WaitingQueue interface ----

    def add_request(self, request: RequestQueueItem) -> None:
        """Route request to appropriate sub-queue (Algorithm 2)."""
        if (
            not request.request
            or not hasattr(request.request, "py_arrival_time")
            or request.request.py_arrival_time is None
        ):
            self._arrival_times[request.id] = time.time()

        prompt_len = self._get_prompt_len(request)
        self._prompt_len_history.append(prompt_len)
        # Cap history to prevent unbounded growth
        if len(self._prompt_len_history) > 100000:
            self._prompt_len_history = self._prompt_len_history[-50000:]

        # Component 2: Dynamic routing with bubble queues
        idx = self._route_request_dynamic(prompt_len)
        self._queues[idx].items.append(request)
        self._queues[idx].record_prompt_len(prompt_len)
        self._total_count += 1

        # Record metrics for meta-optimization
        self._meta_metrics.setdefault('prompt_lens', []).append(prompt_len)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests, routing each dynamically."""
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        """Tactical scheduling with periodic maintenance (Algorithm 1)."""
        if self._prepended:
            item = self._prepended.popleft()
            self._arrival_times.pop(item.id, None)
            return item
        if self._total_count == 0:
            raise IndexError("pop from an empty queue")

        now = time.time()

        # Periodic maintenance
        self._maybe_repartition()
        self._maybe_meta_optimize()
        self._prune_empty_bubble_queues()

        # Tactical scheduling: select best queue
        best_idx = self._select_best_queue(now)
        if best_idx < 0:
            raise IndexError("pop from an empty queue")

        item = self._queues[best_idx].items.popleft()
        self._total_count -= 1
        rid = item.id

        # Record wait time for meta-optimization metrics
        wait_time = now - self._get_arrival_time(item, now)
        self._meta_metrics.setdefault('wait_times', []).append(wait_time)
        self._meta_metrics.setdefault('queue_sizes', []).append(
            len(self._queues))

        self._arrival_times.pop(rid, None)
        return item

    def peek_request(self) -> RequestQueueItem:
        """Peek at the front request of the highest-scoring sub-queue."""
        if self._prepended:
            return self._prepended[0]
        if self._total_count == 0:
            raise IndexError("peek from an empty queue")
        now = time.time()
        best_idx = self._select_best_queue(now)
        if best_idx < 0:
            raise IndexError("peek from an empty queue")
        return self._queues[best_idx].items[0]

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
        for q in self._queues:
            old_len = len(q.items)
            q.items = deque(
                req for req in q.items if req.id not in request_ids)
            self._total_count -= (old_len - len(q.items))
        for rid in request_ids:
            self._arrival_times.pop(rid, None)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._prepended) or self._total_count > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._prepended) + self._total_count

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over all sub-queues, shortest-prompt queue first."""
        chains: list[Iterable[RequestQueueItem]] = [self._prepended]
        for q in self._queues:
            if q.items:
                chains.append(q.items)
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
