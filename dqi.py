from __future__ import annotations

import cmath
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sage.all import GF, vector
from sage.rings.finite_rings.element_base import FiniteRingElement

from decoders import AbstractDecoder, BenchmarkResult

METHODS = ("auto", "analytical", "interference", "average_v_bound")


@dataclass(frozen=True)
class QualityEstimate:
    """What ``estimate_solution_quality`` computed, and how much it is worth.

    ``float(estimate)`` is the objective value; the rest says which estimator ran,
    whether the number is a proven expectation or a sample estimate, and whether
    the observable hit the ``max(0.0, ...)`` floor (a clamped value is m/2, i.e.
    random guessing, and carries no information).

    ``guaranteed`` means the value holds without any measured input (the
    closed form in the perfect-decoding regime).  ``exact`` is weaker and only
    set by the interference estimator: the pair sums are exact, but for the
    benchmark that was measured -- exhaustively (``exact=True``) or on a sample
    of the error patterns of each weight (``exact=False``).
    """

    value: float
    l: int
    method: str
    guaranteed: bool
    epsilon: tuple[float, ...] | None = None
    clamped: bool = False
    exact: bool = False

    def __float__(self) -> float:
        return self.value


def _check_nondegenerate_r(p: int, r: int) -> None:
    if not 0 < r < p:
        raise ValueError(
            f"requires 0 < r < p, got r={r}, p={p}: every F_i is empty (r=0) or "
            f"the whole field (r=p), so the DQI construction does not apply"
        )


class Dqi:
    def __init__(self, instance, decoder_constructor=None):
        self.instance = instance.to_max_linsat()
        if decoder_constructor is None:
            self.decoder_constructor = self.instance.default_decoder_constructor
        else:
            self.decoder_constructor = decoder_constructor
        self.decoder = None
        self.g = None
        self.g_tilde = None
        self.last_estimate: QualityEstimate | None = None

    def get_decoder(self) -> AbstractDecoder:
        if self.decoder is None:
            self.decoder = self.decoder_constructor(self.instance)
        return self.decoder

    def estimate_solution_quality(
        self,
        l: int | None = None,
        *,
        w: np.ndarray | None = None,
        method: str = "auto",
        n_decoding_samples: int | str | None = "auto",
        details: bool = False,
    ) -> float | QualityEstimate:
        """Estimate the objective value DQI reaches on this instance.

        ``l``     how many errors the DQI state mixes in.  Defaults to the largest
                  value that still lies in the perfect-decoding regime, see
                  ``_default_l``.
        ``w``     weight vector of the DQI polynomial, length ``l + 1``.  Defaults
                  to the optimal one, the top eigenvector of A.
        ``method``
                  ``"auto"``            cheapest estimator that is valid here
                  ``"analytical"``      closed form; raises outside that regime
                  ``"interference"``    ``compute_expectation``, for every prime p
                  ``"average_v_bound"`` cheap GF(2) lower bound, often just m/2
                  Both ``"auto"`` and ``"interference"`` take the same
                  ``compute_expectation`` route on GF(2) as on GF(p); the
                  equivalent A_bar formulation lives in ``tests/reference_impl``
                  as a cross-check only.
        ``n_decoding_samples``
                  decoding attempts per error weight, for any field.  ``"auto"``
                  means 500, ``None`` means exhaustive (exponential, but then the
                  estimate needs no population rescaling).
        ``details``
                  return the ``QualityEstimate`` record instead of a bare float.
                  It is also kept in ``self.last_estimate`` eitherway.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")

        field = self.instance.field
        if not field.is_prime_field():
            raise ValueError(
                f"every estimator here is derived over a prime field F_p, got {field}"
            )
        p = field.order()
        _check_nondegenerate_r(p, self.instance.get_r())
        m = self.instance.get_B().nrows()
        is_binary = p == 2

        radius = self.get_decoder().decoding_radius()

        if l is None:
            l = self._default_l(radius)
        l = int(l)
        if not 1 <= l <= m:
            raise ValueError(f"l must satisfy 1 <= l <= m = {m}, got {l}")
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.shape != (l + 1,):
                raise ValueError(
                    f"w must have shape (l + 1,) = ({l + 1},), got {w.shape}"
                )

        # If possible, compute solution quality analytically (without computing decoding benchmarks)
        if (
            method in ("auto", "analytical")
            and radius is not None
            # is decoding perfect
            and l <= radius
            # are semi-circle law requirements fulfilled
            and self._has_exact_A(l, radius)
        ):
            return self._finish(
                self._analytical_estimate(l, w),
                l,
                "analytical",
                guaranteed=True,
                details=details,
            )

        # requirements for analytical estimate not met
        if method == "analytical":
            raise ValueError(
                f"l = {l} is not in the perfect-decoding regime: decoding radius "
                f"{radius}, minimum distance {self._minimum_distance()} (needs "
                f'l <= radius and 2l + 2 <= d). method="auto" measures the decoding '
                f"failure rates instead of assuming they are zero"
            )

        # sampled benchmarks are rescaled to the full population by
        # compute_expectation, so they are allowed for every field
        n_tries = self._resolve_n_decoding_samples(n_decoding_samples)
        benchmarks = self._benchmarks(l, n_tries)
        epsilon = [benchmarks[k].epsilon for k in range(l + 1)]

        # When no error was incorrectly decoded, assume perfect decoding regime
        if method == "auto" and max(epsilon) == 0 and self._has_exact_A(l, None):
            return self._finish(
                self._analytical_estimate(l, w),
                l,
                "analytical",
                guaranteed=n_tries is None,
                epsilon=epsilon,
                details=details,
            )

        if method == "average_v_bound":
            if not is_binary:
                raise ValueError(
                    "the average-v bound is only derived for GF(2); use "
                    'method="interference"'
                )
            value = self._dqi_lower_bound_average_v(l, w=w, benchmarks=benchmarks)
            clamped = value <= m / 2
            if clamped:
                warnings.warn(
                    f"the observable was clamped to 0, so the bound degenerated to "
                    f"m/2 = {m / 2} (random guessing); try a smaller l or "
                    'method="interference"',
                    stacklevel=2,
                )
            return self._finish(
                value,
                l,
                "average_v_bound",
                guaranteed=False,
                epsilon=epsilon,
                clamped=clamped,
                details=details,
            )

        return self._finish(
            compute_expectation(self.instance, l, w, benchmarks),
            l,
            "interference",
            guaranteed=False,
            exact=n_tries is None,
            epsilon=epsilon,
            details=details,
        )

    def _default_l(self, radius: int | None) -> int:
        """Largest l where semi-circle law applies.

        Semi-circle law requires:
            2l + 1 < d <=>
            2l + 2 <= d <=>
            l <= d / 2 - 1 <=>
            l <= floor(d / 2 - 1) = d // 2 - 1  (since l is an integer)
        """
        l = self._minimum_distance() // 2 - 1
        if radius is not None:
            # cap l at the maximum value where the decoder guarantees perfect decoding
            l = min(l, radius)
        # if l = 0, fall back to l = 1 (then imperfect regime)
        return max(1, min(l, self.instance.get_B().nrows()))

    def _has_exact_A(self, l: int, radius: int | None) -> bool:
        """
        Check requirements for semi-circle law
        """

        # radius <= (d - 1) // 2 by the unique decoding threshold
        # radius <= (d - 1) // 2 <= (d - 1) / 2 <=> 2 radius <= d - 1
        #
        # Therefore, if l < radius <=> l + 1 <= radius, then
        # 2l + 2 <= 2 radius <= d - 1 < d
        # Thus, 2l + 1 < 2l + l < d: The requirment of the semi-circle law is met.
        # We can therefore skip computing the minimum distance
        if radius is not None and l < radius:
            return True
        return 2 * l + 2 <= self._minimum_distance()

    def _minimum_distance(self) -> int:
        return self.instance.get_minimum_distance()

    def _resolve_n_decoding_samples(
        self, n_decoding_samples: int | str | None
    ) -> int | None:
        """``"auto"`` is 500 decoding attempts per error weight, on any field."""
        if n_decoding_samples == "auto":
            return 500
        return n_decoding_samples

    def _benchmarks(self, l: int, n_tries: int | None) -> dict[int, BenchmarkResult]:
        """Benchmark every error weight 1..l, keyed by weight (weight 0 never fails)."""
        decoder = self.get_decoder()
        m = self.instance.get_B().nrows()
        benchmarks = {0: BenchmarkResult([vector(self.instance.field, [0] * m)], [], 0)}
        # Starting at large errors is advantageous for some decoders
        for k in reversed(range(1, l + 1)):
            benchmarks[k] = decoder.get_benchmarks(l, k, n_tries=n_tries)
        return benchmarks

    def _analytical_estimate(self, l: int, w: np.ndarray | None) -> float:
        p = self.instance.field.order()
        r = self.instance.get_r()
        m = self.instance.get_B().nrows()
        if w is None:
            return _predict_dqi_performance_perfect_optimal_w(p, r, m, l)
        return _predict_dqi_performance_perfect(p, r, m, l, w)

    def _finish(
        self,
        value: float,
        l: int,
        method: str,
        *,
        guaranteed: bool,
        details: bool,
        epsilon: list[float] | None = None,
        clamped: bool = False,
        exact: bool = False,
    ) -> float | QualityEstimate:
        estimate = QualityEstimate(
            value=float(value),
            l=l,
            method=method,
            guaranteed=guaranteed,
            epsilon=None if epsilon is None else tuple(float(e) for e in epsilon),
            clamped=clamped,
            exact=exact,
        )
        self.last_estimate = estimate
        return estimate if details else estimate.value

    def semicircle_law_solution_quality(
        self, l: int | None = None, w: np.ndarray | None = None
    ) -> float:
        """DQI performance in the perfect-decoding regime (the semicircle law).

        Valid as long as the decoder corrects every error of weight <= l and
        2l + 2 <= d, which is what ``estimate_solution_quality`` checks before it
        takes this route.  The default l is the largest one satisfying the second
        condition; note that the unique-decoding radius (d - 1) // 2 is one too
        large for odd d, where a weight-d codeword still reaches A_bar[l, l].
        """
        d = self._minimum_distance()
        if l is None:
            l = d // 2 - 1
        l = int(l)
        if l < 1:
            raise ValueError(
                f"minimum distance d = {d} admits no l >= 1 with 2l + 2 <= d, so the "
                f"perfect-decoding regime is empty for this instance"
            )
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.shape != (l + 1,):
                raise ValueError(
                    f"w must have shape (l + 1,) = ({l + 1},), got {w.shape}"
                )
        if 2 * l + 2 > d:
            warnings.warn(
                f"2l + 2 = {2 * l + 2} > d = {d}: A_bar is not exactly A for l = {l}, "
                f"so the returned value is not guaranteed",
                stacklevel=2,
            )
        return self._analytical_estimate(l, w)

    def _dqi_lower_bound_average_v(
        self,
        l: int,
        w: np.ndarray | None = None,
        n_tries=500,
        benchmarks: dict[int, BenchmarkResult] | None = None,
    ):
        assert self.instance.field == GF(2) and self.instance.get_r() == 1

        if benchmarks is None:
            benchmarks = self._benchmarks(l, n_tries)
        # epsilon indexed by error weight.
        # decoding zero errors always succeeds
        epsilon = [benchmarks[k].epsilon for k in range(l + 1)]

        m = self.instance.get_B().nrows()

        return _upper_bound_dqi_performance_imperfect_average_v(m, l, epsilon, w)

    def _compute_g_g_tilde(self):
        """The objective indicators g_i and their Fourier transforms g~_i, as callables.

        The estimators use the (m, p) table ``_g_tilde_table`` directly; these
        closures are the same values with the per-element interface.
        """
        if not self.instance.field.is_prime_field():
            raise ValueError("This function only supports prime fields")
        p = self.instance.field.order()
        r = self.instance.get_r()
        _check_nondegenerate_r(p, r)
        f_dash = 2 * r / p - 1
        phi = math.sqrt(4 * r * (1 - r / p))

        yes_value = (+1 - f_dash) / phi
        no_value = (-1 - f_dash) / phi

        def make_g_lambda(F_i):
            return lambda y: yes_value if y in F_i else no_value

        self.g = [make_g_lambda(F_i) for F_i in self.instance.get_F()]

        table = _g_tilde_table(p, r, self.instance.get_F())

        def make_g_tilde_lambda(g_tilde_i):
            return lambda y: complex(g_tilde_i[int(y) % p])

        self.g_tilde = [make_g_tilde_lambda(row) for row in table]

    def _get_g_tilde(self) -> list[Callable[[FiniteRingElement], complex]]:
        if self.g is None:
            self._compute_g_g_tilde()
        return list(self.g_tilde)


def get_eigenvector(mat: np.ndarray, index: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(mat)
    value = values[index]
    vector = vectors[:, index]
    return value, vector


def make_A(p: int, r: int, m: int, l: int) -> np.array:
    _check_nondegenerate_r(p, r)

    diag = (p - 2 * r) / math.sqrt(r * (p - r)) * np.diag(np.arange(0, l + 1))

    k = np.arange(0, l + 1)
    a_k = np.sqrt(k * (m - k + 1))

    off_diag = np.diag(a_k)
    return diag + np.roll(off_diag, -1, 0) + np.roll(off_diag, -1, 1)


def _predict_dqi_performance_perfect(p: int, r: int, m: int, l: int, w: int) -> float:
    A = make_A(p, r, m, l)

    assert len(w) == l + 1
    norm = np.linalg.norm(w)

    polynomial_effect = (
        w.reshape((1, l + 1)) @ A @ w.reshape((l + 1, 1)) / (norm * norm)
    )

    return ((m * r) / p + np.sqrt(r * (p - r)) / p * polynomial_effect).item()


def _predict_dqi_performance_perfect_optimal_w(p: int, r: int, m: int, l: int) -> float:
    A = make_A(p, r, m, l)

    lambda_1 = np.linalg.eigh(A)[0][-1]
    return (m * r) / p + np.sqrt(r * (p - r)) / p * lambda_1


def _upper_bound_dqi_performance_imperfect_average_v(
    m: int, l: int, epsilon: float, w: np.ndarray = None
) -> float:
    if isinstance(epsilon, float) or isinstance(epsilon, int):
        epsilon = [float(epsilon)] * (l + 1)

    assert len(epsilon) == l + 1
    assert min(epsilon) >= 0
    assert w is None or len(w) == l + 1

    A = make_A(2, 1, m, l)

    if w is None:
        lambda_1, w = get_eigenvector(A, -1)
        matrix_product = lambda_1
    else:
        w = np.array(w)
        matrix_product = w.reshape((1, l + 1)) @ A @ w.reshape((l + 1, 1))

    largest_epsilon = max(epsilon)
    epsilon = np.array(epsilon)

    norm = np.sum(w * w * (1 - epsilon))

    observable = (matrix_product - np.sum(w * w) * 2 * largest_epsilon * (m + 1)) / norm

    return (max(0.0, observable.item()) + m) / 2


# --------------------------------------------------------------------------- #
# Imperfect-decoding estimators.
#
# The estimator sums over *pairs* of decodable errors (y1, y2).  The pair
# condition is a syndrome shift,
#
#     B^T (y1 - y2 + a e_i) = 0   <=>   s(y1) = s(y2) - a * b_i,
#
# with s(y) = B^T y the syndrome and b_i = B^T e_i the i-th column of B^T (the
# i-th row of B).  So instead of enumerating pairs, we compute every syndrome
# once, put the decodable errors into a dict keyed by their syndrome and find
# the partner of each error with one dict lookup.  That is O(m p N) lookups
# instead of O(l^2 N^2 m p) Sage operations.
# --------------------------------------------------------------------------- #


def _error_matrix(vectors, m: int) -> np.ndarray:
    """Stack Sage vectors (or anything iterable of ints) into an (N, m) int array."""
    if len(vectors) == 0:
        return np.zeros((0, m), dtype=np.int64)
    return np.array([[int(c) for c in v] for v in vectors], dtype=np.int64)


def _syndrome_dtype(p: int) -> np.dtype:
    """Smallest unsigned dtype holding [0, p): the bytes of a row in it are its dict key."""
    return np.min_scalar_type(p - 1)


def _lookup_rows(lookup: dict[bytes, int], rows: np.ndarray, p: int) -> np.ndarray:
    """Global index of every syndrome row in ``rows`` (..., n); -1 where none is decodable."""
    n = rows.shape[-1]
    buf = np.ascontiguousarray(rows, dtype=_syndrome_dtype(p)).tobytes()
    step = n * _syndrome_dtype(p).itemsize
    idx = np.fromiter(
        (lookup.get(buf[j : j + step], -1) for j in range(0, len(buf), step)),
        dtype=np.int64,
        count=rows.size // n,
    )
    return idx.reshape(rows.shape[:-1])


def _g_tilde_table(p: int, r: int, F) -> np.ndarray:
    """(m, p) complex table g_tilde[i, a] = 1/sqrt(p) sum_x omega^{a x} g_i(x).

    The Fourier transform of the normalised objective indicator g_i (mean 0,
    sum_x g_i(x)^2 = 1), evaluated eagerly for every field element instead of
    re-running the O(p) Fourier sum on every lookup.
    """
    _check_nondegenerate_r(p, r)
    f_dash = 2 * r / p - 1
    phi = math.sqrt(4 * r * (1 - r / p))
    yes, no = (1 - f_dash) / phi, (-1 - f_dash) / phi
    omega = cmath.exp(2j * math.pi / p)
    m = len(F)
    g = np.full((m, p), no, dtype=float)
    for i, F_i in enumerate(F):
        for f in F_i:
            g[i, int(f)] = yes
    a = np.arange(p)
    fourier = omega ** np.outer(a, a) / math.sqrt(p)  # fourier[a, x]
    return g @ fourier.T


def _amplitudes(
    Y: np.ndarray, gt: np.ndarray, w: np.ndarray, k: int, m: int
) -> np.ndarray:
    """DQI amplitudes c(y) = w_k G(y) / sqrt(C(m, k)) of one weight-k error block.

    G(y) = prod_{i: y_i != 0} g~_i(y_i), the coefficient of |B^T y> in the DQI
    state (g~_i(0) does not appear, hence the ``where``).
    """
    gathered = gt[np.arange(m)[None, :], Y]  # (N, m)
    G = np.where(Y == 0, 1.0 + 0j, gathered).prod(axis=1)
    return w[k] * G / math.sqrt(math.comb(m, k))


def _syndrome_index(
    benchmarks: dict[int, BenchmarkResult], l: int, Bnp: np.ndarray, p: int
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[bytes, int]]:
    """Errors and syndromes per error weight plus the syndrome dict, with the E7 check.

    Returns ``({k: (Y, S)}, lookup)`` with ``Y`` the (N_k, m) decodable weight-k
    errors, ``S = Y B mod p`` their (N_k, n) syndromes (in ``_syndrome_dtype``)
    and ``lookup`` mapping the bytes of a syndrome row to the *global* index of
    its error, i.e. its position in the concatenation of the blocks in weight
    order (``_lookup_rows`` does the probing).

    The DQI circuit decodes a *syndrome*, so its decoder maps every syndrome to
    exactly one error: a valid benchmark therefore never reports two correct
    errors with the same syndrome.  If it does, the pair sums below would model
    a state in which several errors of one coset interfere coherently, which is
    not the state DQI prepares -- so this is rejected rather than silently
    estimated.
    """
    blocks, lookup, seen = {}, {}, []
    for k in range(l + 1):
        Y = _error_matrix(benchmarks[k].correct, Bnp.shape[0])
        S = ((Y @ Bnp) % p).astype(_syndrome_dtype(p))
        for j, row in enumerate(S):
            key = row.tobytes()
            if key in lookup:
                k_other, y_other = seen[lookup[key]]
                raise ValueError(
                    f"invalid benchmark: the errors "
                    f"{[(k_other, y_other), (k, Y[j].tolist())]} "
                    f"(as (weight, error)) share the syndrome {row.tolist()}, "
                    f"but a syndrome decoder returns exactly one error per "
                    f"syndrome, so at most one of them can be decoded correctly"
                )
            lookup[key] = len(seen)
            seen.append((k, Y[j].tolist()))
        blocks[k] = (Y, S)
    return blocks, lookup


def compute_expectation(
    instance,
    l: int,
    w: np.ndarray | None,
    benchmarks: dict[int, BenchmarkResult],
) -> float:
    """Expected number of satisfied constraints of the DQI state, any prime p.

    The state is the one DQI actually prepares, restricted to the errors the
    decoder gets right:  |psi> ~ sum_k w_k / sqrt(C(m,k)) sum_{y in C_k} G(y) |B^T y>
    with G(y) = prod_{i: y_i != 0} g~_i(y_i).  Its objective value is

        <s> = 1/p sum_i sum_a phase[i, a] sum_{y1, y2: s(y1) = s(y2) - a b_i}
                  conj(c(y1)) c(y2)  /  sum_y |c(y)|^2

    with phase[i, a] = sum_{u in F_i} omega^{-a u}.  Exact given the benchmarks.

    Sampled benchmarks (``n_decoding_samples`` not ``None``) cover only part of
    each weight class, so every weight-k block is rescaled to the full
    population: there are ``C(m,k) (p-1)^k`` errors of weight k, a fraction
    ``1 - epsilon_k`` of which is decodable, so a block of ``N_k`` sampled
    errors stands for ``scale_k = C(m,k) (p-1)^k (1 - epsilon_k) / N_k`` times as
    much amplitude weight.  A pair (y1, y2) of *distinct* errors from the blocks
    (k1, k2) therefore gets ``scale_k1 * scale_k2``, but a term with y1 = y2 is a
    single draw and gets ``scale_k1``, as does a norm term; for exhaustive
    benchmarks every ``scale_k`` is 1.  This is why the errors are bucketed per
    weight pair instead of pooled: pooling would apply one common factor to all
    weights.
    """
    field = instance.field
    if not field.is_prime_field():
        raise ValueError("This function only supports prime fields")
    p = field.order()
    r = instance.get_r()
    m = instance.get_m()
    Bnp = np.array(instance.get_B(), dtype=np.int64)
    n = Bnp.shape[1]
    F = instance.get_F()

    if w is None:
        _, w = get_eigenvector(make_A(p, r, m, l), -1)
        w = w / np.linalg.norm(w)
    w = np.asarray(w, dtype=float)
    if w.shape != (l + 1,):
        raise ValueError(f"w must have shape (l + 1,) = ({l + 1},), got {w.shape}")

    gt = _g_tilde_table(p, r, F)  # (m, p)
    omega = cmath.exp(2j * math.pi / p)

    # F_i Fourier phases: phase[i, a] = sum_{u in F_i} omega^{-a u}
    a_range = np.arange(p)
    phase = np.zeros((m, p), dtype=complex)
    for i, F_i in enumerate(F):
        for u in F_i:
            phase[i] += omega ** (-a_range * int(u))

    blocks, lookup = _syndrome_index(benchmarks, l, Bnp, p)

    S, c, scale = {}, {}, {}
    for k in range(l + 1):
        Y, S[k] = blocks[k]
        c[k] = _amplitudes(Y, gt, w, k, m)
        # rescale the sampled block to the full weight-k population
        N_k = max(1, len(Y))  # an empty block has no amplitude to scale
        scale[k] = math.comb(m, k) * (p - 1) ** k * (1 - benchmarks[k].epsilon) / N_k
    # amplitude and scale by global index, the index space of ``lookup``
    c_all = np.concatenate([c[k] for k in range(l + 1)])
    scale_all = np.concatenate([np.full(len(c[k]), scale[k]) for k in range(l + 1)])
    start = np.cumsum([0] + [len(c[k]) for k in range(l)])

    norm = float(np.sum(scale_all * np.abs(c_all) ** 2))
    if norm == 0:
        raise ValueError(
            "no decodable errors in the benchmarks: the DQI state is empty"
        )

    expectation = 0j
    for k1 in range(l + 1):
        n_1 = len(c[k1])
        if n_1 == 0:
            continue
        # a = 0: no shift, so y2 must have the syndrome of y1, i.e. y2 = y1
        expectation += scale[k1] * phase[:, 0].sum() * np.sum(np.abs(c[k1]) ** 2)
        own = (start[k1] + np.arange(n_1))[None, :]
        # keep the (chunk, N_k1, n) shift array below at a few million entries
        chunk = max(1, min(m, 4_000_000 // max(1, n_1 * n)))
        for a in range(1, p):
            # for every y1 look up the decodable error with syndrome s(y1) + a b_i
            for lo in range(0, m, chunk):
                b = Bnp[lo : lo + chunk]  # the rows b_i of this chunk
                idx = _lookup_rows(
                    lookup, (S[k1][None, :, :] + a * b[:, None, :]) % p, p
                )
                hit = idx >= 0  # (chunk, N_k1)
                # a pair of *different* errors stands for scale_k1 * scale_k2 pairs
                # of the population, a term y2 = y1 (only if a b_i = 0) for scale_k1
                # single errors
                weight = np.where(idx == own, scale[k1], scale[k1] * scale_all[idx])
                term = phase[lo : lo + chunk, a][:, None] * (
                    np.conj(c[k1])[None, :] * c_all[idx] * weight
                )
                expectation += np.sum(np.where(hit, term, 0))

    return float((expectation / p / norm).real)
