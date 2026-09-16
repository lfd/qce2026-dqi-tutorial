from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Iterator

import numpy as np
import scipy as sp
from ldpc import BpDecoder, BpOsdDecoder
from sage.all import GF, vector
from sage.coding.decoder import DecodingError
from sage.coding.grs_code import GeneralizedReedSolomonCode, GRSBerlekampWelchDecoder
from sage.coding.guruswami_sudan.gs_decoder import GRSGuruswamiSudanDecoder
from sage.coding.information_set_decoder import LinearCodeInformationSetDecoder
from sage.coding.linear_code import (
    LinearCodeNearestNeighborDecoder,
    LinearCodeSyndromeDecoder,
)


class BenchmarkResult:
    def __init__(self, correct: list, incorrect: list, epsilon: float):
        self.correct = correct
        self.incorrect = incorrect
        self.epsilon = epsilon
        for e in correct:
            e.set_immutable()
        for e in incorrect:
            e.set_immutable()
        self.correct_set = set(correct)
        self.incorrect_set = set(incorrect)

    def __str__(self) -> str:
        return f"BenchmarkResult(#correct = {len(self.correct)}, #incorrect = {len(self.incorrect)}, epsilon = {self.epsilon})"


def generate_all_errors(field: GF, n: int, k: int) -> Iterator[vector]:
    els = [e for e in field if e != 0]

    for error_indices in (
        np.array(comb) for comb in itertools.combinations(range(n), k)
    ):
        for error_digits in itertools.product(els, repeat=k):
            e = vector(field, [0] * n)
            for i, idx in enumerate(error_indices):
                e[idx] += error_digits[i]
            yield e


def generate_errors(field: GF, n: int, k: int, n_tries: int | None) -> Iterator[vector]:
    els = [e for e in field if e != 0]

    n_possible_errors = math.comb(n, k) * len(els) ** k
    if n_tries is None or n_possible_errors <= n_tries:
        for e in generate_all_errors(field, n, k):
            yield e

    elif 2 * n_tries >= n_possible_errors:
        all_errors = [e for e in generate_all_errors(field, n, k)]
        random.shuffle(all_errors)
        for i in range(n_tries):
            yield all_errors[i]
    else:
        seen = set()
        n_produced = 0
        while n_produced < n_tries:
            e = vector(field, [0] * n)
            error_indices = np.random.choice(n, k, replace=False)
            for i in error_indices:
                e[i] += random.choice(els)
            e.set_immutable()
            if e in seen:
                continue
            seen.add(e)
            n_produced += 1
            yield e


class AbstractDecoder:
    """Base class of all decoders.

    A subclass implements *exactly one* of the two decoding interfaces:

    * ``decode_syndrome(s, l)`` -- the natural interface for DQI: the decoder
      only sees the syndrome ``s = H e`` (with ``H = B^T`` the parity check
      matrix of the DQI code ``ker(B^T)``) and returns an error estimate of
      length ``m``, or ``None`` if decoding fails.
    * ``decode_codeword(y, l)`` -- the classical word based interface: given a
      received word ``y`` of length ``m`` return the nearest codeword.

    Word based decoders are turned into syndrome decoders by
    :meth:`error_estimate` via a fixed coset representative ``rho(s)`` with
    ``H rho(s) = s``: the estimate is ``rho - decode_codeword(rho)``.  This is
    what the DQI circuit does, and in particular it returns exactly one error
    per syndrome (a word based decoder fed the error itself breaks ties per
    word and would report several errors of the same coset as correct).
    """

    def __init__(self, instance: MaxLinSat):
        self.instance = instance
        self.benchmarks = {}

        cls = type(self)
        has_syndrome = cls.decode_syndrome is not AbstractDecoder.decode_syndrome
        has_codeword = cls.decode_codeword is not AbstractDecoder.decode_codeword
        if has_syndrome and has_codeword:
            raise TypeError(
                f"{cls.__name__} overrides both decode_syndrome and decode_codeword; "
                "a decoder must implement exactly one of them"
            )
        if not has_syndrome and not has_codeword:
            raise TypeError(
                f"{cls.__name__} implements neither decode_syndrome nor decode_codeword; "
                "a decoder must implement exactly one of them"
            )
        self.decodes_syndromes = has_syndrome

        # H = B^T is the parity check matrix of the code ker(B^T) used by DQI.
        self.H = self.instance.get_B().T
        self._estimate_cache = {}
        if has_codeword:
            self._setup_representative()

    def _setup_representative(self):
        """Precompute a linear map s -> rho(s) with H rho(s) == s."""
        H = self.H

        # eliminate redundant parity-checks
        R = list(H.T.pivots())
        H_R = H.matrix_from_rows(R)

        # select a basis for the syndrome space
        P = list(H_R.pivots())
        self._rep_rows = R
        self._rep_cols = P
        self._rep_inverse = H_R.matrix_from_columns(P).inverse()

    def representative(self, s: vector) -> vector:
        """A fixed word rho of length m with H rho == s (linear in s)."""
        field = self.instance.field
        rho = vector(field, [0] * self.instance.get_m())
        if self._rep_cols:
            values = self._rep_inverse * vector(field, [s[i] for i in self._rep_rows])
            for j, col in enumerate(self._rep_cols):
                rho[col] = values[j]
        return rho

    def decoding_radius(self) -> int | None:
        raise NotImplementedError()

    def _complete_decoding_radius(self) -> int | None:
        """
        Returns how many errors would be returned by a "complete" decoder, that is one which (by excaustive search or similar) always decodes to the nearest codeword.

        Complete decoders can use this as their decoding_radius implementation.
        """
        d = self.instance.get_minimum_distance()
        if d <= 0:
            return None
        return (d - 1) // 2

    def decode_syndrome(self, s: vector, l: int) -> vector | None:
        """Return an error estimate for the syndrome s, or None on failure."""
        raise NotImplementedError()

    def decode_codeword(self, y: vector, l: int) -> vector:
        """Return the codeword closest to the received word y."""
        raise NotImplementedError()

    def error_estimate(self, s: vector, l: int) -> vector | None:
        """The unique error estimate the decoder assigns to the syndrome s.

        Returns ``None`` if the decoder fails on ``s``.
        """
        s.set_immutable()
        if self.decodes_syndromes:
            return self.decode_syndrome(s, l)

        # if the decoder is a codeword decoder:
        # 1. compute syndrome representative rho
        # 2. decode rho -> c
        # 3. report rho - c
        key = (l, s)
        if key not in self._estimate_cache:
            rho = self.representative(s)
            try:
                codeword = self.decode_codeword(rho, l)
            except DecodingError:
                estimate = None
            else:
                estimate = rho - codeword
                estimate.set_immutable()
            self._estimate_cache[key] = estimate
        return self._estimate_cache[key]

    def compute_benchmarks(self, l: int, n_errors: int, n_tries: int | None):
        correct = list()
        incorrect = list()

        m = self.instance.get_m()

        for e in generate_errors(self.instance.field, m, n_errors, n_tries):
            s = self.H * e
            estimate = self.error_estimate(s, l)

            if estimate is not None and estimate == e:
                correct.append(e)
            else:
                incorrect.append(e)

        n_total = len(correct) + len(incorrect)
        if n_total == 0:
            raise ValueError(
                f"No errors of weight {n_errors} exist for a code of length {m}"
            )
        epsilon = len(incorrect) / n_total

        self.benchmarks[(l, n_errors, n_tries)] = BenchmarkResult(
            correct, incorrect, epsilon
        )

    def get_benchmarks(
        self, l: int, n_errors: int, n_tries: int | None = 100
    ) -> BenchmarkResult:
        key = (l, n_errors, n_tries)
        if key not in self.benchmarks:
            self.compute_benchmarks(l, n_errors, n_tries)
        return self.benchmarks[key]


class NearestNeighborDecoder(AbstractDecoder):
    @staticmethod
    def constructor() -> Callable[[MaxLinSat], AbstractDecoder]:
        return NearestNeighborDecoder

    def __init__(self, instance: MaxLinSat):
        super().__init__(instance)
        self.decoder = LinearCodeNearestNeighborDecoder(instance.get_code())

    def decoding_radius(self) -> int | None:
        return self._complete_decoding_radius()

    def decode_codeword(self, y: vector, l: int) -> vector:
        return self.decoder.decode_to_code(y)


class SyndromeDecoder(AbstractDecoder):
    @staticmethod
    def constructor(**decoder_parameters) -> Callable[[MaxLinSat], AbstractDecoder]:
        return lambda instance: SyndromeDecoder(instance, **decoder_parameters)

    def __init__(self, instance: MaxLinSat, **decoder_parameters):
        super().__init__(instance)
        # accepted for interface compatibility, LinearCodeSyndromeDecoder has no options
        self.decoder_parameters = decoder_parameters
        self.decoders = {}
        self.syndrome_tables = {}

    def get_decoder(self, l: int) -> LinearCodeSyndromeDecoder:
        valid_ls = [k for k in self.decoders if k >= l]

        if len(valid_ls) > 0:
            return self.decoders[min(valid_ls)]

        decoder = LinearCodeSyndromeDecoder(self.instance.get_code(), l)
        self.decoders[l] = decoder
        # The syndrome decoder is based on code.parity_check_matrix(),
        # which can differ from B^T.
        # We must therefore create our own table
        table = {}
        for err in decoder.syndrome_table().values():
            e = vector(self.instance.field, err)
            e.set_immutable()
            s = self.H * e
            s.set_immutable()
            table[s] = e
        self.syndrome_tables[l] = table
        return decoder

    def get_syndrome_table(self, l: int) -> dict:
        valid_ls = [k for k in self.syndrome_tables if k >= l]
        if len(valid_ls) > 0:
            return self.syndrome_tables[min(valid_ls)]
        self.get_decoder(l)
        return self.syndrome_tables[l]

    def decoding_radius(self) -> int | None:
        return self._complete_decoding_radius()

    def decode_syndrome(self, s: vector, l: int) -> vector | None:
        s.set_immutable()
        return self.get_syndrome_table(l).get(s)


class InformationSetDecoder(AbstractDecoder):
    @staticmethod
    def constructor(**decoder_parameters) -> Callable[[MaxLinSat], AbstractDecoder]:
        return lambda instance: InformationSetDecoder(instance, **decoder_parameters)

    def __init__(self, instance: MaxLinSat, **decoder_parameters):
        super().__init__(instance)
        self.decoder_parameters = decoder_parameters
        self.decoders = {}

    def get_decoder(self, l: int) -> LinearCodeInformationSetDecoder:
        if l not in self.decoders:
            self.decoders[l] = LinearCodeInformationSetDecoder(
                self.instance.get_code(), l, **self.decoder_parameters
            )
        return self.decoders[l]

    def decoding_radius(self):
        return None

    def decode_codeword(self, y: vector, l: int) -> vector:
        return self.get_decoder(l).decode_to_code(y)


class BeliefPropagationDecoder(AbstractDecoder):
    @staticmethod
    def constructor(**decoder_parameters) -> Callable[[MaxLinSat], AbstractDecoder]:
        return lambda instance: BeliefPropagationDecoder(instance, **decoder_parameters)

    def __init__(self, instance: MaxLinSat, **decoder_parameters):
        super().__init__(instance)
        # the ldpc decoder must use the same parity check matrix (and hence the
        # same syndrome basis) as the framework: H = B^T
        self.H_sparse = sp.sparse.csc_matrix(np.array(self.H).astype(np.int8))
        self.decoder_parameters = decoder_parameters
        self.decoders = {}

    def decoding_radius(self) -> None:
        return None

    def get_decoder(self, l: int) -> BpDecoder:
        if l not in self.decoders:
            error_rate = l / self.instance.get_m()
            parameters = dict(self.decoder_parameters)
            parameters["input_vector_type"] = "syndrome"
            self.decoders[l] = BpDecoder(
                self.H_sparse, error_rate=error_rate, **parameters
            )
        return self.decoders[l]

    def decode_syndrome(self, s: vector, l: int) -> vector:
        syndrome_np = np.array([int(x) for x in s], dtype=np.uint8)
        decoder = self.get_decoder(l)
        result_np = decoder.decode(syndrome_np)
        return vector(self.instance.field, result_np)


class BeliefPropagationOsdDecoder(AbstractDecoder):
    @staticmethod
    def constructor(**decoder_parameters) -> Callable[[MaxLinSat], AbstractDecoder]:
        return lambda instance: BeliefPropagationOsdDecoder(
            instance, **decoder_parameters
        )

    def decoding_radius(self) -> None:
        return None

    def __init__(self, instance: MaxLinSat, **decoder_parameters):
        super().__init__(instance)
        self.H_sparse = sp.sparse.csc_matrix(np.array(self.H).astype(np.int8))
        self.decoder_parameters = decoder_parameters
        self.decoders = {}

    def get_decoder(self, l: int) -> BpOsdDecoder:
        if l not in self.decoders:
            error_rate = l / self.instance.get_m()
            self.decoders[l] = BpOsdDecoder(
                self.H_sparse, error_rate=error_rate, **self.decoder_parameters
            )
        return self.decoders[l]

    def decode_syndrome(self, s: vector, l: int) -> vector:
        syndrome_np = np.array([int(x) for x in s], dtype=np.uint8)
        decoder = self.get_decoder(l)
        result_np = decoder.decode(syndrome_np)
        return vector(self.instance.field, result_np)


class GeneralizedReedSolomonDecoder(AbstractDecoder):
    @staticmethod
    def constructor() -> Callable[[MaxLinSat], AbstractDecoder]:
        return GeneralizedReedSolomonDecoder

    def __init__(self, instance: MaxLinSat):
        super().__init__(instance)
        code = self.instance.get_code()
        if not isinstance(code, GeneralizedReedSolomonCode):
            raise TypeError(
                f"GeneralizedReedSolomonDecoder needs a GRS code, got {type(code).__name__}"
            )
        self.code = code
        self.decoder = GRSBerlekampWelchDecoder(code)
        self.list_decoders = {}

    def get_decoder_by_l(
        self, l: int
    ) -> GRSBerlekampWelchDecoder | GRSGuruswamiSudanDecoder:
        radius = self.decoding_radius()
        if l <= radius:
            return self.decoder
        if l not in self.list_decoders:
            # use unique decoder as fallback
            decoder = self.decoder
            for tau in range(l, radius, -1):
                # Using a decoding radius beyond the theoretical list-decoding limit (Johnson bound) fails
                try:
                    decoder = GRSGuruswamiSudanDecoder(self.code, tau)
                    break
                except ValueError:
                    continue
            self.list_decoders[l] = decoder
        return self.list_decoders[l]

    def decoding_radius(self) -> int:
        return (self.instance.get_minimum_distance() - 1) // 2

    def decode_codeword(self, y: vector, l: int) -> vector:
        try:
            result = self.get_decoder_by_l(l).decode_to_code(y)
        except ValueError as ex:
            # Workaround for a (potential) Sage bug:
            # The Berlekamp-Welch decoder can raise a ValueError instead of a DecodingError when given a word beyond its radius
            raise DecodingError(str(ex)) from ex
        if isinstance(result, list):
            # The list decoder producing an empty list is a decoding error
            if len(result) == 0:
                raise DecodingError("list decoding returned no codeword")
            return min(result, key=lambda c: (y - c).hamming_weight())
        return result


def DEFAULT_DECODER_CONSTRUCTOR(instance):
    if instance.field.order() == 2:
        return BeliefPropagationDecoder(instance)
    return InformationSetDecoder(instance)
