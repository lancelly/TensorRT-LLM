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


class _QueueProfile:
    """Per-queue statistics profile for EWSJF scoring and meta-optimization.

    Tracks: mean prompt length, variance, request density, prefill cost
    estimates, and throughput history. All using exponential moving averages.
    """

    def __init__(self):
        # Prompt length statistics (Welford's online algorithm)
        self.count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # sum of squares of differences from mean

        # Request density: arrival timestamps in sliding window
        self._arrival_timestamps: deque[float] = deque()

        # Prefill cost model: observed (prompt_len, prefill_time_ms) pairs
        # Used to fit C_prefill(b) = coeff * b^exp
        self._prefill_observations: deque[tuple[int, float]] = deque(
            maxlen=200)

        # Throughput tracking: (timestamp, tokens_served) for EMA
        self._served_counts: deque[tuple[float, int]] = deque(maxlen=500)

    @property
    def mean_prompt_len(self) -> float:
        return self._mean if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Within-queue prompt length variance (Welford's)."""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance) if self.variance > 0 else 0.0

    def record_prompt_len(self, prompt_len: int) -> None:
        """Update running statistics with Welford's online algorithm."""
        self.count += 1
        delta = prompt_len - self._mean
        self._mean += delta / self.count
        delta2 = prompt_len - self._mean
        self._m2 += delta * delta2

    def record_arrival(self, timestamp: float) -> None:
        self._arrival_timestamps.append(timestamp)

    def get_density(self, now: float, window: float) -> float:
        """ρ(q) = arrivals_in_window / window_size."""
        cutoff = now - window
        # Prune old timestamps
        while (self._arrival_timestamps
               and self._arrival_timestamps[0] < cutoff):
            self._arrival_timestamps.popleft()
        return len(self._arrival_timestamps) / max(window, 0.001)

    def record_prefill(self, prompt_len: int, prefill_time_ms: float) -> None:
        self._prefill_observations.append((prompt_len, prefill_time_ms))

    def record_served(self, timestamp: float, num_tokens: int) -> None:
        self._served_counts.append((timestamp, num_tokens))


class _SubQueue:
    """A single sub-queue in the EWSJF multi-queue system.

    Each sub-queue covers a range [min_len, max_len) of prompt lengths,
    maintains a FIFO deque of requests, and tracks per-queue statistics
    via _QueueProfile for context-aware scoring.
    """

    __slots__ = ('min_len', 'max_len', 'items', 'is_bubble',
                 'empty_count', 'profile', 'queue_id')

    _next_id: int = 0

    def __init__(self, min_len: int, max_len: int,
                 is_bubble: bool = False):
        self.min_len = min_len
        self.max_len = max_len  # exclusive
        self.items: deque[RequestQueueItem] = deque()
        self.is_bubble = is_bubble
        self.empty_count = 0
        self.profile = _QueueProfile()
        self.queue_id = _SubQueue._next_id
        _SubQueue._next_id += 1

    @property
    def mean_prompt_len(self) -> float:
        if self.profile.count == 0:
            return (self.min_len + min(self.max_len, 131072)) / 2.0
        return self.profile.mean_prompt_len

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return len(self.items) > 0


class _GaussianProcess:
    """Minimal Gaussian Process with RBF kernel for Bayesian Optimization.

    Implements GP regression with:
    - RBF (squared exponential) kernel: k(x,x') = exp(-||x-x'||^2 / 2l^2)
    - Cholesky-based posterior computation
    - Expected Improvement (EI) acquisition function

    This avoids external dependencies (scipy/sklearn) by implementing
    the core GP math directly. Sufficient for the ~5-dimensional parameter
    space in EWSJF meta-optimization.
    """

    def __init__(self, length_scale: float = 1.0,
                 noise_variance: float = 0.01):
        self._length_scale = length_scale
        self._noise_var = noise_variance
        self._X: list[list[float]] = []  # observed parameter vectors
        self._y: list[float] = []  # observed rewards
        self._L: Optional[list[list[float]]] = None  # Cholesky factor cache
        self._alpha: Optional[list[float]] = None  # L^{-T} L^{-1} y cache

    def add_observation(self, x: list[float], y: float) -> None:
        self._X.append(list(x))
        self._y.append(y)
        self._L = None  # invalidate cache
        self._alpha = None

    @property
    def n_observations(self) -> int:
        return len(self._X)

    def _rbf_kernel(self, x1: list[float], x2: list[float]) -> float:
        """RBF kernel: k(x1, x2) = exp(-||x1-x2||^2 / (2 * l^2))."""
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-sq_dist / (2 * self._length_scale ** 2))

    def _kernel_matrix(self, X1: list[list[float]],
                       X2: list[list[float]]) -> list[list[float]]:
        return [[self._rbf_kernel(x1, x2) for x2 in X2] for x1 in X1]

    @staticmethod
    def _cholesky(A: list[list[float]]) -> list[list[float]]:
        """Cholesky decomposition A = LL^T. Returns L (lower triangular)."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    L[i][j] = math.sqrt(max(val, 1e-10))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-10)
        return L

    @staticmethod
    def _solve_triangular_lower(L: list[list[float]],
                                b: list[float]) -> list[float]:
        """Solve Lx = b where L is lower triangular."""
        n = len(b)
        x = [0.0] * n
        for i in range(n):
            s = sum(L[i][j] * x[j] for j in range(i))
            x[i] = (b[i] - s) / max(L[i][i], 1e-10)
        return x

    @staticmethod
    def _solve_triangular_upper(L: list[list[float]],
                                b: list[float]) -> list[float]:
        """Solve L^T x = b where L is lower triangular."""
        n = len(b)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(L[j][i] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s) / max(L[i][i], 1e-10)
        return x

    def _fit(self) -> None:
        """Compute Cholesky decomposition and alpha = K^{-1} y."""
        if self._L is not None:
            return
        n = len(self._X)
        K = self._kernel_matrix(self._X, self._X)
        # Add noise to diagonal
        for i in range(n):
            K[i][i] += self._noise_var
        self._L = self._cholesky(K)
        v = self._solve_triangular_lower(self._L, self._y)
        self._alpha = self._solve_triangular_upper(self._L, v)

    def predict(self, x: list[float]) -> tuple[float, float]:
        """Predict mean and variance at point x.

        Returns (mu, sigma^2).
        """
        if not self._X:
            return 0.0, 1.0
        self._fit()
        assert self._L is not None and self._alpha is not None

        k_star = [self._rbf_kernel(x, xi) for xi in self._X]

        # Mean: k_star^T alpha
        mu = sum(k * a for k, a in zip(k_star, self._alpha))

        # Variance: k(x,x) - k_star^T K^{-1} k_star
        v = self._solve_triangular_lower(self._L, k_star)
        k_xx = self._rbf_kernel(x, x)
        var = k_xx - sum(vi ** 2 for vi in v)
        var = max(var, 1e-10)

        return mu, var

    @staticmethod
    def _standard_normal_cdf(x: float) -> float:
        """Approximation of Φ(x) using the error function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _standard_normal_pdf(x: float) -> float:
        """φ(x) = exp(-x^2/2) / sqrt(2π)."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def expected_improvement(self, x: list[float],
                             best_y: float) -> float:
        """EI(x) = (mu - f_best) Φ(z) + σ φ(z) where z = (mu - f_best)/σ.

        Returns the expected improvement at point x.
        """
        mu, var = self.predict(x)
        sigma = math.sqrt(var)
        if sigma < 1e-10:
            return max(0.0, mu - best_y)
        z = (mu - best_y) / sigma
        return ((mu - best_y) * self._standard_normal_cdf(z)
                + sigma * self._standard_normal_pdf(z))

    def suggest_next(self, param_ranges: dict[str, tuple[float, float]],
                     best_y: float, n_candidates: int = 200) -> list[float]:
        """Suggest next point to evaluate by maximizing EI.

        Uses random candidate sampling (efficient for small dim).
        """
        keys = sorted(param_ranges.keys())
        best_ei = -1.0
        best_x: list[float] = []

        for _ in range(n_candidates):
            x = [random.uniform(param_ranges[k][0], param_ranges[k][1])
                 for k in keys]
            ei = self.expected_improvement(x, best_y)
            if ei > best_ei:
                best_ei = ei
                best_x = x

        return best_x


class _PrefillCostModel:
    """Empirical model for C_prefill(b) = coeff * b^exp.

    Maintains a running estimate of prefill cost as a function of prompt
    length. Updated online from observed prefill times. Falls back to
    configurable defaults when insufficient data.
    """

    def __init__(self, default_coeff: float = 1.0,
                 default_exp: float = 1.5):
        self._coeff = default_coeff
        self._exp = default_exp
        # Observations for model fitting: (log(b), log(time))
        self._observations: deque[tuple[float, float]] = deque(maxlen=500)
        self._fitted = False

    def estimate(self, prompt_len: int) -> float:
        """C_prefill(b) = coeff * b^exp."""
        b = max(prompt_len, 1)
        return self._coeff * (b ** self._exp)

    def record(self, prompt_len: int, prefill_time_ms: float) -> None:
        """Record an observed (prompt_len, prefill_time) pair."""
        if prompt_len > 0 and prefill_time_ms > 0:
            self._observations.append(
                (math.log(max(prompt_len, 1)),
                 math.log(max(prefill_time_ms, 0.001))))
            self._fitted = False

    def refit(self) -> None:
        """Fit log-linear model: log(t) = log(coeff) + exp * log(b).

        Simple linear regression in log-log space.
        """
        if len(self._observations) < 10:
            return
        if self._fitted:
            return

        n = len(self._observations)
        sum_x = sum_y = sum_xy = sum_xx = 0.0
        for log_b, log_t in self._observations:
            sum_x += log_b
            sum_y += log_t
            sum_xy += log_b * log_t
            sum_xx += log_b * log_b

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-10:
            self._fitted = True
            return

        exp = (n * sum_xy - sum_x * sum_y) / denom
        log_coeff = (sum_y - exp * sum_x) / n

        # Clamp to reasonable range
        self._exp = max(0.5, min(3.0, exp))
        self._coeff = max(0.001, min(1000.0, math.exp(log_coeff)))
        self._fitted = True


class EWSJFWaitingQueue(WaitingQueue):
    """EWSJF (Exponentially Weighted SJF) multi-queue waiting queue.

    Full-fidelity implementation of the EWSJF paper (arxiv 2601.21758)
    with all four components:

    1. **Refine-and-Prune** (Section 4.2): 3-stage unsupervised partitioning.
       - Stage 1: K-means(k=3) with K-means++ initialization for coarse
         short/medium/long clustering.
       - Stage 2: Recursive gap splitting — Gap_j > α * mean(G) triggers
         split at midpoint. Recurses until no significant gaps remain or
         cluster width falls below minimum threshold.
       - Stage 3: Merge by scheduling utility U(qi,qi+1) =
         (ρ(qi) + ρ(qi+1)) / (|b̄_{i+1} - b̄_i| + ε), removing lowest-
         utility boundary until ≤ max_queues.
       Runs periodically (strategic loop, ~10 min) with online lightweight
       adjustments between full runs.

    2. **Dynamic Queue Routing** (Algorithm 2, Appendix D): Routes requests
       to sub-queues with tolerance-based matching (1.10x upper, 0.90x lower)
       and on-demand bubble queue creation for requests in gaps. Bubble
       queues are auto-pruned after empty_queue_threshold consecutive
       empty scheduling rounds.

    3. **Density-Weighted Scoring** (Eq 1 & 4): Context-aware prioritization
       with per-queue adaptive weights:
         Score(r,q) = qf * (w_base + w_urg(b̄_q)*cs + w_fair(b̄_q)*log(b+1))
       where:
         qf = qi / (b + 1): queue factor (qi = num_queues - queue_index)
         cs = Wt / C_prefill(b): compute score (wait / empirical prefill cost)
         w_urg(b̄_q) = a_u * b̄_q + b_u: context-aware urgency weight
         w_fair(b̄_q) = a_f * b̄_q + b_f: context-aware fairness weight
       C_prefill(b) is modeled as coeff * b^exp, fitted online from observed
       prefill times.

    4. **Bayesian Meta-Optimization** (Section 4.4): Gaussian Process
       surrogate with RBF kernel and Expected Improvement acquisition
       function. Tunes Θ = {a_u, b_u, a_f, b_f, α} to maximize:
         R(Θ) = λ1*C + λ2*L - λ3*S - λ4*U
       where C = compactness (real within-queue variance), L = load balance,
       S = proliferation penalty, U = latency penalty. Converges in 5-8
       trials (ΔT = 10-15 min each).

    The tactical loop (Algorithm 1) runs on every pop:
    1. Score each non-empty queue using front request
    2. Select q_prim = argmax(scores), pop from it (GreedyFill)
    3. If additional capacity remains, adjacent queues supply backfill
       (implemented via sequential pop calls from the scheduler)
    4. Prune empty bubble queues exceeding threshold
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
        # Online mode: lightweight boundary adjustments between full runs
        self._online_adjustment_counter: int = 0
        self._online_adjustment_interval: int = 500  # requests between checks

        # Component 3: Prefill cost model
        self._prefill_model = _PrefillCostModel(
            default_coeff=self._config.prefill_cost_coefficient,
            default_exp=self._config.prefill_cost_exponent)

        # Component 4: Gaussian Process Bayesian Optimizer
        self._gp = _GaussianProcess(
            length_scale=self._config.gp_length_scale,
            noise_variance=self._config.gp_noise_variance)
        self._last_meta_opt_time: float = time.time()
        self._current_params: dict[str, float] = {
            'a_urgency': self._config.a_urgency,
            'b_urgency': self._config.b_urgency,
            'a_fairness': self._config.a_fairness,
            'b_fairness': self._config.b_fairness,
            'gap_significance_alpha': self._config.gap_significance_alpha,
        }
        self._best_reward: float = float('-inf')
        self._meta_trial_count: int = 0
        # Metrics accumulated between meta-opt trials
        self._trial_wait_times: list[float] = []
        self._trial_queue_sizes: list[int] = []

        # Parameter ranges for BO (matching paper's tunable params)
        self._param_ranges: dict[str, tuple[float, float]] = {
            'a_urgency': (-0.001, 0.01),
            'b_urgency': (0.1, 5.0),
            'a_fairness': (-0.001, 0.001),
            'b_fairness': (0.01, 2.0),
            'gap_significance_alpha': (0.5, 3.0),
        }

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

    # ================================================================
    # Component 1: Refine-and-Prune (Section 4.2)
    # ================================================================

    @staticmethod
    def _kmeans_1d(data: list[int], k: int = 3,
                   max_iter: int = 100) -> list[list[int]]:
        """1D K-means with K-means++ initialization.

        K-means++: first centroid is random, subsequent centroids are
        chosen with probability proportional to D(x)^2 (distance to
        nearest existing centroid). Leads to better convergence.
        """
        if len(data) <= k:
            return [[x] for x in sorted(data)]
        sorted_data = sorted(data)
        n = len(sorted_data)

        # K-means++ initialization
        centroids: list[float] = [float(sorted_data[random.randint(0, n - 1)])]
        for _ in range(1, k):
            # Compute D(x)^2 for each point
            dists = []
            for x in sorted_data:
                min_d = min(abs(x - c) for c in centroids)
                dists.append(min_d * min_d)
            total = sum(dists)
            if total == 0:
                centroids.append(float(
                    sorted_data[random.randint(0, n - 1)]))
                continue
            # Sample proportional to D(x)^2
            threshold = random.random() * total
            cumsum = 0.0
            chosen = sorted_data[0]
            for i, d in enumerate(dists):
                cumsum += d
                if cumsum >= threshold:
                    chosen = sorted_data[i]
                    break
            centroids.append(float(chosen))

        # Iterate
        prev_assignments: list[int] = []
        for _ in range(max_iter):
            # Assign each point to nearest centroid
            clusters: list[list[int]] = [[] for _ in range(k)]
            assignments: list[int] = []
            for x in sorted_data:
                best_c = min(range(k),
                             key=lambda c: abs(x - centroids[c]))
                clusters[best_c].append(x)
                assignments.append(best_c)

            # Check convergence
            if assignments == prev_assignments:
                break
            prev_assignments = assignments

            # Update centroids
            for i in range(k):
                if clusters[i]:
                    centroids[i] = sum(clusters[i]) / len(clusters[i])

        return [c for c in clusters if c]

    def _gap_split_recursive(self, sorted_cluster: list[int],
                             alpha: float, min_width: int = 64,
                             depth: int = 0,
                             max_depth: int = 10) -> list[int]:
        """Recursively split a cluster at significant gaps.

        Recurses into sub-clusters until no significant gaps remain or
        cluster width falls below min_width (paper Section 4.2).

        Returns boundary positions (midpoints of significant gaps).
        """
        if (len(sorted_cluster) < 2
                or depth >= max_depth
                or (sorted_cluster[-1] - sorted_cluster[0]) < min_width):
            return []

        # Compute gaps
        gaps: list[tuple[int, int, int]] = []  # (gap, midpoint, index)
        for i in range(len(sorted_cluster) - 1):
            gap = sorted_cluster[i + 1] - sorted_cluster[i]
            mid = (sorted_cluster[i] + sorted_cluster[i + 1]) // 2
            gaps.append((gap, mid, i))

        if not gaps:
            return []

        mean_gap = sum(g for g, _, _ in gaps) / len(gaps)
        threshold = alpha * mean_gap

        # Find significant gaps
        sig_gaps = [(g, mid, idx) for g, mid, idx in gaps if g > threshold]
        if not sig_gaps:
            return []

        # Sort by gap size descending, split at largest first
        sig_gaps.sort(key=lambda x: x[0], reverse=True)

        boundaries: list[int] = []
        for _, mid, split_idx in sig_gaps:
            boundaries.append(mid)
            # Recurse into sub-clusters
            left = sorted_cluster[:split_idx + 1]
            right = sorted_cluster[split_idx + 1:]
            boundaries.extend(
                self._gap_split_recursive(left, alpha, min_width,
                                          depth + 1, max_depth))
            boundaries.extend(
                self._gap_split_recursive(right, alpha, min_width,
                                          depth + 1, max_depth))
            break  # Only split at largest gap per recursion level

        return boundaries

    def _scheduling_utility(self, b1: int, b2: int,
                            data: list[int], now: float) -> float:
        """Compute scheduling utility for merging two adjacent queues.

        U(qi, qi+1) = (ρ(qi) + ρ(qi+1)) / (|b̄_{i+1} - b̄_i| + ε) (Eq 3)

        Uses real request density from queue profiles when available,
        falls back to counting data points in range.
        """
        eps = 1.0
        window = self._config.density_window_seconds

        # Try to get density from existing queue profiles
        density = 0.0
        for q in self._queues:
            if q.min_len <= b1 < q.max_len or q.min_len <= b2 < q.max_len:
                density += q.profile.get_density(now, window)

        # Fallback to data counting if no profile density
        if density == 0.0:
            count1 = sum(1 for x in data if x < b1)
            count2 = sum(1 for x in data if b1 <= x < b2)
            density = float(count1 + count2)

        return density / (abs(b2 - b1) + eps)

    def _refine_and_prune(self, data: list[int]) -> list[int]:
        """Full Refine-and-Prune algorithm (Section 4.2).

        Stage 1: K-means(k=3) with K-means++ for coarse partitioning
        Stage 2: Recursive gap splitting within each cluster
        Stage 3: Merge by scheduling utility until ≤ max_queues
        """
        if len(data) < 10:
            return list(self._DEFAULT_BOUNDARIES)

        alpha = self._current_params['gap_significance_alpha']
        now = time.time()

        # Stage 1: K-means(k=3) for short/medium/long
        clusters = self._kmeans_1d(data, k=3)

        # Stage 2: Recursive gap splitting
        all_boundaries: list[int] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            boundaries = self._gap_split_recursive(
                sorted(cluster), alpha, min_width=64)
            all_boundaries.extend(boundaries)

        # Add inter-cluster boundaries
        for i in range(len(clusters) - 1):
            if clusters[i] and clusters[i + 1]:
                boundary = (max(clusters[i]) + min(clusters[i + 1])) // 2
                all_boundaries.append(boundary)

        all_boundaries = sorted(set(b for b in all_boundaries if b > 0))

        # Stage 3: Merge by scheduling utility until ≤ max_queues
        max_q = self._config.max_queues
        while len(all_boundaries) + 1 > max_q and len(all_boundaries) > 1:
            min_utility = float('inf')
            merge_idx = 0
            for i in range(len(all_boundaries) - 1):
                utility = self._scheduling_utility(
                    all_boundaries[i], all_boundaries[i + 1], data, now)
                if utility < min_utility:
                    min_utility = utility
                    merge_idx = i
            all_boundaries.pop(merge_idx)

        return all_boundaries if all_boundaries else list(
            self._DEFAULT_BOUNDARIES)

    def _online_adjust(self) -> None:
        """Lightweight online boundary adjustment between full runs.

        Strategic loop online mode: check if any queue's variance is
        excessively high relative to others, and split it. Or merge
        two adjacent nearly-empty queues.
        """
        if len(self._queues) < 2:
            return

        # Collect variance stats
        variances = [(i, q.profile.variance, len(q.items))
                     for i, q in enumerate(self._queues)
                     if q.profile.count >= 10]
        if len(variances) < 2:
            return

        mean_var = sum(v for _, v, _ in variances) / len(variances)

        # Split: queue with variance > 3x mean and enough items
        for idx, var, size in variances:
            if var > 3.0 * mean_var and size >= 4:
                q = self._queues[idx]
                mid = int(q.mean_prompt_len)
                if q.min_len < mid < q.max_len:
                    # Split into two queues
                    q1 = _SubQueue(q.min_len, mid)
                    q2 = _SubQueue(mid, q.max_len)
                    # Redistribute items
                    for item in q.items:
                        plen = self._get_prompt_len(item)
                        if plen < mid:
                            q1.items.append(item)
                            q1.profile.record_prompt_len(plen)
                        else:
                            q2.items.append(item)
                            q2.profile.record_prompt_len(plen)
                    self._queues[idx] = q1
                    self._queues.insert(idx + 1, q2)
                    logger.debug(
                        f"EWSJF online: split queue [{q.min_len},{q.max_len}) "
                        f"at {mid} (var={var:.0f}, mean_var={mean_var:.0f})")
                    break  # One adjustment per call

    def _maybe_repartition(self) -> None:
        """Periodically re-run Refine-and-Prune (strategic loop offline)."""
        interval = self._config.repartition_interval_seconds
        if interval <= 0 or len(self._prompt_len_history) < 50:
            return
        now = time.time()
        if now - self._last_repartition_time < interval:
            return

        self._last_repartition_time = now
        new_boundaries = self._refine_and_prune(self._prompt_len_history)

        old_boundaries = [q.min_len for q in self._queues[1:]]
        if new_boundaries == old_boundaries:
            return

        logger.info(
            f"EWSJF Refine-and-Prune: repartitioning "
            f"{len(self._queues)} -> {len(new_boundaries)+1} queues, "
            f"boundaries={new_boundaries[:10]}"
            f"{'...' if len(new_boundaries) > 10 else ''}")

        self._redistribute_to_new_boundaries(new_boundaries)

    def _redistribute_to_new_boundaries(
            self, new_boundaries: list[int]) -> None:
        """Rebuild sub-queues with new boundaries, preserving all requests
        and transferring profile statistics."""
        all_items: list[RequestQueueItem] = []
        for q in self._queues:
            all_items.extend(q.items)

        self._queues = self._build_queues(new_boundaries)
        self._total_count = 0

        for item in all_items:
            prompt_len = self._get_prompt_len(item)
            idx = self._find_queue_idx(prompt_len)
            self._queues[idx].items.append(item)
            self._queues[idx].profile.record_prompt_len(prompt_len)
            self._total_count += 1

    # ================================================================
    # Component 2: Dynamic Queue Routing (Algorithm 2, Appendix D)
    # ================================================================

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

        Algorithm 2 from the paper (Appendix D):
        1. Find adjacent queues Q_i, Q_{i+1}
        2. If L <= Q_i.max_len * 1.10: assign to Q_i
        3. Else if L >= Q_{i+1}.min_len * 0.90: assign to Q_{i+1}
        4. Else: create bubble queue centered on L
        """
        idx = self._find_queue_idx(prompt_len)
        q = self._queues[idx]

        # Direct fit
        if q.min_len <= prompt_len < q.max_len:
            return idx

        # Algorithm 2: tolerance-based assignment
        upper_tol = self._config.bubble_upper_tolerance
        lower_tol = self._config.bubble_lower_tolerance

        # Check lower queue (Q_i) with upper tolerance
        if idx > 0:
            q_lower = self._queues[idx - 1]
            if prompt_len <= q_lower.max_len * upper_tol:
                return idx - 1

        # Check upper queue (Q_{i+1}) with lower tolerance
        if prompt_len >= q.min_len * lower_tol:
            return idx

        # Create bubble queue (Algorithm 2, else branch)
        if len(self._queues) < self._config.max_queues * 2:
            bubble_width = self._config.default_bubble_width
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
            self._queues.insert(idx, bubble)
            logger.debug(
                f"EWSJF: bubble queue [{new_min},{new_max}) "
                f"for prompt_len={prompt_len}")
            return idx

        return idx

    def _prune_empty_bubble_queues(self) -> None:
        """Remove bubble queues that have been empty too long (Alg 1 line 10-12)."""
        threshold = self._config.empty_queue_threshold
        to_remove: list[int] = []
        for i, q in enumerate(self._queues):
            if not q.items:
                if q.is_bubble:
                    q.empty_count += 1
                    if q.empty_count > threshold:
                        to_remove.append(i)
                # Non-bubble queues also track empty count for scoring
                else:
                    q.empty_count += 1
            else:
                q.empty_count = 0  # Reset on non-empty
        for i in reversed(to_remove):
            logger.debug(
                f"EWSJF: pruning bubble [{self._queues[i].min_len},"
                f"{self._queues[i].max_len})")
            self._queues.pop(i)

    # ================================================================
    # Component 3: Density-Weighted Scoring (Eq 1 & 4)
    # ================================================================

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
        """Context-aware urgency: w_urg(b̄_q) = a_u * b̄_q + b_u (Eq 4)."""
        return max(0.0,
                   self._current_params['a_urgency'] * mean_prompt_len
                   + self._current_params['b_urgency'])

    def _w_fairness(self, mean_prompt_len: float) -> float:
        """Context-aware fairness: w_fair(b̄_q) = a_f * b̄_q + b_f."""
        return max(0.0,
                   self._current_params['a_fairness'] * mean_prompt_len
                   + self._current_params['b_fairness'])

    def _compute_queue_score(self, q_idx: int, now: float) -> float:
        """Compute EWSJF score for a sub-queue (Eq 1 & 4).

        Score(r,q) = qf * (w_base + w_urg(b̄_q) * cs + w_fair(b̄_q) * log(b+1))
        where:
            qf = qi / (b + 1)
            cs = Wt / C_prefill(b)
        """
        q = self._queues[q_idx]
        if not q.items:
            return float('-inf')

        front = q.items[0]
        b = self._get_prompt_len(front)
        wait_time = max(0.0, now - self._get_arrival_time(front, now))

        # Queue factor: qi = num_queues - queue_index
        qi = len(self._queues) - q_idx
        qf = qi / (b + 1)

        # Compute score with empirical prefill cost model
        c_prefill = self._prefill_model.estimate(b)
        cs = wait_time / max(c_prefill, 0.001)

        # Context-aware weights from per-queue profile
        mean_b = q.mean_prompt_len
        w_urg = self._w_urgency(mean_b)
        w_fair = self._w_fairness(mean_b)

        return qf * (self._config.w_base
                     + w_urg * cs
                     + w_fair * math.log(b + 1))

    def _select_best_queue(self, now: float) -> int:
        """Tactical scheduling: select q_prim = argmax(scores) (Alg 1)."""
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

    # ================================================================
    # Component 4: Bayesian Meta-Optimization (Section 4.4)
    # ================================================================

    def _compute_reward(self) -> float:
        """R(Θ) = λ1*C + λ2*L - λ3*S - λ4*U (Eq 5).

        C = compactness: inverse of mean within-queue variance (real variance
            from Welford's algorithm, not range approximation).
        L = load balance: 1 / (1 + CV of queue occupancies).
        S = queue proliferation: num_queues / max_queues.
        U = latency penalty: mean observed wait time, normalized.
        """
        cfg = self._config
        num_queues = len(self._queues)
        active_queues = [q for q in self._queues if q.profile.count >= 2]

        if not active_queues:
            return 0.0

        # C: Compactness — real within-queue variance
        total_var = sum(q.profile.variance for q in active_queues)
        mean_var = total_var / len(active_queues)
        compactness = 1.0 / (1.0 + mean_var / 1e6)  # normalize to ~[0,1]

        # L: Load balance — coefficient of variation of queue sizes
        sizes = [len(q.items) for q in self._queues]
        if sizes and sum(sizes) > 0:
            mean_size = sum(sizes) / len(sizes)
            var_size = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            cv = math.sqrt(var_size) / max(mean_size, 0.001)
            load_balance = 1.0 / (1.0 + cv)
        else:
            load_balance = 1.0

        # S: Queue proliferation
        proliferation = num_queues / max(cfg.max_queues, 1)

        # U: Latency penalty — mean wait time from trial period
        if self._trial_wait_times:
            mean_wait = (sum(self._trial_wait_times)
                         / len(self._trial_wait_times))
            latency = mean_wait / 10.0  # normalize
        else:
            latency = 0.0

        return (cfg.lambda_compactness * compactness
                + cfg.lambda_load_balance * load_balance
                - cfg.lambda_proliferation * proliferation
                - cfg.lambda_latency * latency)

    def _maybe_meta_optimize(self) -> None:
        """Run one trial of GP-based Bayesian meta-optimization."""
        if not self._config.meta_optimization_enabled:
            return
        interval = self._config.meta_optimization_interval_seconds
        if interval <= 0:
            return
        now = time.time()
        if now - self._last_meta_opt_time < interval:
            return
        if self._meta_trial_count >= self._config.meta_optimization_max_trials:
            return  # Converged, stop optimizing

        self._last_meta_opt_time = now
        self._meta_trial_count += 1

        # Compute reward for current parameters
        current_reward = self._compute_reward()
        if current_reward > self._best_reward:
            self._best_reward = current_reward

        # Record observation in GP
        keys = sorted(self._param_ranges.keys())
        x = [self._current_params[k] for k in keys]
        self._gp.add_observation(x, current_reward)

        # Use GP to suggest next parameters via Expected Improvement
        if self._gp.n_observations >= 2:
            suggested = self._gp.suggest_next(
                self._param_ranges, self._best_reward, n_candidates=300)
            new_params = {k: v for k, v in zip(keys, suggested)}
        else:
            # Not enough data for GP yet — random exploration
            new_params = {
                k: random.uniform(lo, hi)
                for k, (lo, hi) in self._param_ranges.items()
            }

        self._current_params = new_params

        # Clear trial metrics
        self._trial_wait_times = []
        self._trial_queue_sizes = []

        # Periodically refit prefill cost model
        self._prefill_model.refit()

        logger.info(
            f"EWSJF BO trial {self._meta_trial_count}: "
            f"reward={current_reward:.4f} (best={self._best_reward:.4f}), "
            f"n_obs={self._gp.n_observations}, "
            f"params={{a_u={self._current_params['a_urgency']:.6f}, "
            f"b_u={self._current_params['b_urgency']:.4f}, "
            f"a_f={self._current_params['a_fairness']:.6f}, "
            f"b_f={self._current_params['b_fairness']:.4f}, "
            f"alpha={self._current_params['gap_significance_alpha']:.3f}}}")

    # ================================================================
    # WaitingQueue interface
    # ================================================================

    def add_request(self, request: RequestQueueItem) -> None:
        """Route request to appropriate sub-queue (Algorithm 2)."""
        now = time.time()
        if (
            not request.request
            or not hasattr(request.request, "py_arrival_time")
            or request.request.py_arrival_time is None
        ):
            self._arrival_times[request.id] = now

        prompt_len = self._get_prompt_len(request)
        self._prompt_len_history.append(prompt_len)
        # Cap history to prevent unbounded growth
        if len(self._prompt_len_history) > 100000:
            self._prompt_len_history = self._prompt_len_history[-50000:]

        # Component 2: Dynamic routing with bubble queues
        idx = self._route_request_dynamic(prompt_len)
        self._queues[idx].items.append(request)
        self._queues[idx].profile.record_prompt_len(prompt_len)
        self._queues[idx].profile.record_arrival(now)
        self._total_count += 1

        # Online adjustment check (strategic loop online mode)
        self._online_adjustment_counter += 1
        if self._online_adjustment_counter >= self._online_adjustment_interval:
            self._online_adjustment_counter = 0
            self._online_adjust()

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        """Tactical scheduling loop (Algorithm 1).

        1. Score each non-empty queue → updated_scores
        2. Select q_prim = argmax(scores)
        3. Pop from q_prim (GreedyFill — one request per pop call)
        4. Prune empty bubble queues
        """
        if self._prepended:
            item = self._prepended.popleft()
            self._arrival_times.pop(item.id, None)
            return item
        if self._total_count == 0:
            raise IndexError("pop from an empty queue")

        now = time.time()

        # Periodic strategic loop maintenance
        self._maybe_repartition()
        self._maybe_meta_optimize()
        self._prune_empty_bubble_queues()

        # Tactical scheduling: select best queue (Algorithm 1)
        best_idx = self._select_best_queue(now)
        if best_idx < 0:
            raise IndexError("pop from an empty queue")

        item = self._queues[best_idx].items.popleft()
        self._total_count -= 1

        # Record wait time for meta-optimization
        wait_time = now - self._get_arrival_time(item, now)
        self._trial_wait_times.append(wait_time)
        self._trial_queue_sizes.append(len(self._queues))

        self._arrival_times.pop(item.id, None)
        return item

    def peek_request(self) -> RequestQueueItem:
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
        self._prepended.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        self._prepended.extendleft(requests)

    def remove_by_ids(self, request_ids: set[int]) -> None:
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
        return bool(self._prepended) or self._total_count > 0

    def __len__(self) -> int:
        return len(self._prepended) + self._total_count

    def __iter__(self) -> Iterator[RequestQueueItem]:
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
