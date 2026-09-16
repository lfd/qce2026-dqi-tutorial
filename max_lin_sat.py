from __future__ import annotations

import math
import random
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from enum import Enum

import ldpc.code_util
import networkx as nx
import numpy as np
import scipy as sp
from sage.all import GF, Matrix, codes, next_prime, vector
from sage.coding.grs_code import GeneralizedReedSolomonCode
from sage.coding.linear_code import AbstractLinearCode
from sage.libs.gap.util import GAPError
from sage.rings.finite_rings.element_base import FiniteRingElement

from classical_solvers import (
    BruteForceSolver,
    OrToolsSolver,
)
from constraints import LinSatConstraint, LinSatVar
from decoders import (
    DEFAULT_DECODER_CONSTRUCTOR,
    AbstractDecoder,
    BeliefPropagationDecoder,
    GeneralizedReedSolomonDecoder,
    InformationSetDecoder,
    NearestNeighborDecoder,
)
from gadget import Gadget


class AbstractMaxLinSat:
    def __init__(
        self,
        field: GF,
        variables: list[LinSatVar],
        default_decoder_constructor: Callable[[AbstractMaxLinSat], AbstractDecoder],
    ):
        self.field = field
        self.variables = variables
        self.optimal_solution = None
        self.default_decoder_constructor = default_decoder_constructor
        self.decoder = None
        self.optimal_solution = None

    def to_max_linsat(self) -> AbstractMaxLinSat:
        return self

    def get_B(self) -> Matrix:
        raise NotImplementedError()

    def get_n(self) -> int:
        return self.get_B().ncols()

    def get_m(self) -> int:
        return self.get_B().nrows()

    def get_F(self) -> list[set[FiniteRingElement]]:
        raise NotImplementedError()

    def get_weights(self) -> list[int]:
        return None

    def get_code(self) -> AbstractLinearCode:
        raise NotImplementedError()

    def get_minimum_distance(self) -> int:
        raise NotImplementedError()

    def evaluate_solution(self, x: np.ndarray) -> int:
        B = self.get_B()
        F = self.get_F()
        weights = self.get_weights()
        x = vector(self.field, x)
        y = B * x

        if weights is None:
            return sum((1 if y_i in F_i else 0 for y_i, F_i in zip(y, F)))
        else:
            return sum(
                (w_i if y_i in F_i else 0 for y_i, F_i, w_i in zip(y, F, weights))
            )

    def get_v(self) -> vector:
        F = self.get_F()
        if not all(len(f) == 1 for f in self.F):
            raise ValueError("v is not well-defined since not all F_i have a size of 1")
        return vector(self.field, [next(iter(f)) for f in self.F])

    def get_r(self) -> int:
        F = self.get_F()
        r = None
        for F_i in F:
            if r is None:
                r = len(F_i)
            else:
                if len(F_i) != r:
                    raise ValueError("The F_i are not of the same size")
        return r

    def compute_optimal_solution(self):
        solver = OrToolsSolver(self)
        solution = solver.get_solution()
        if solver.is_optimal():
            self.optimal_solution = solution
        else:
            brute_force_solver = BruteForceSolver(self)
            self.optimal_solution = brute_force_solver.get_solution()

    def get_optimal_solution(self) -> vector:
        if self.optimal_solution is None:
            self.compute_optimal_solution()
        return self.optimal_solution

    def get_optimal_solution_quality(self) -> int:
        return self.evaluate_solution(self.get_optimal_solution())

    def get_random_solution_value(self) -> int:
        return sum(len(F_i) / len(self.field) for F_i in self.F)


class MergeStrategy(Enum):
    WEIGHTS = 0
    DUPLICATES = 1
    USE_STRICTEST = 2
    USE_LOOSEST = 3


class MaxLinSat(AbstractMaxLinSat):
    def __init__(
        self,
        field: GF,
        default_decoder_constructor=DEFAULT_DECODER_CONSTRUCTOR,
        merge_strategy=MergeStrategy.DUPLICATES,
        equal_size_F_i=True,
    ):
        super().__init__(
            field=field,
            variables=[],
            default_decoder_constructor=default_decoder_constructor,
        )
        self.constraints = {}

        self.B = None
        self.F = None
        self.weights = None
        self.code = None
        self.minimum_distance = None
        self.merge_strategy = merge_strategy
        self.equal_size_F_i = equal_size_F_i

    def clear_cache(self):
        self.B = None
        self.F = None
        self.weights = None
        self.code = None
        self.minimum_distance = None
        self.optimal_solution = None

    def new_var(self, name: str | None = None) -> LinSatVar:
        var = LinSatVar(self.field, len(self.variables), name)
        self.variables.append(var)
        self.clear_cache()
        return var

    def add_constraint(
        self,
        constraint: LinSatConstraint,
        weight: int | None = None,
        disable_warnings=False,
    ):
        if constraint.field != self.field:
            raise ValueError("Field mismatch", self.field, constraint.field)
        for var in constraint.vars:
            if var.id not in self.variables or var.name != self.variables[var.id].name:
                raise ValueError("Unknown variable", var)

        if not disable_warnings:
            constraint.show_degeneracy_warnings()

        if weight is not None:
            constraint = constraint.scale_weight(weight)

        cid = constraint.get_id()
        if cid not in self.constraints:
            self.constraints[cid] = constraint
        else:
            self.constraints[cid] = self.constraints[cid].merge(constraint)
        self.clear_cache()

    def add_gadget(self, gadget: Gadget, *variables: LinSatVar):
        for b_i, F_i in zip(gadget.B, gadget.F):
            self.add_constraint(LinSatConstraint(self.field, variables, b_i, F_i))

    def set_equal_size_F_i(self, value):
        self.equal_size_F_i = value

    def compute_B_F_weights(self):
        constraints = [c for c in self.constraints.values() if not c.is_degenerate]
        constraints.sort(key=lambda c: c.get_id())

        gcd = 0
        for c in constraints:
            for weight in c.rhs.values():
                gcd = math.gcd(gcd, weight)

        if gcd != 0:
            for c in constraints:
                for el, weight in c.rhs.items():
                    c.rhs[el] = weight // gcd

        used_variables = set()

        B_rows = []
        F = []
        weights = []
        for constraint in constraints:
            row = [self.field(0)] * len(self.variables)
            for coef, var in zip(constraint.coefs, constraint.vars):
                row[var.id] += coef
                used_variables.add(var.id)

            rhs_by_weight = {}
            for el, weight in constraint.rhs.items():
                if weight not in rhs_by_weight:
                    rhs_by_weight[weight] = set()
                rhs_by_weight[weight].add(el)

            if len(rhs_by_weight) == len(self.field):
                minimum = min(rhs_by_weight.keys())
                new_rhs_by_weight = {}
                for weight, values in rhs_by_weight.items():
                    if weight == minimum:
                        continue
                    new_rhs_by_weight[weight - minimum] = values
                rhs_by_weight = new_rhs_by_weight

            sorted_weights = sorted(list(rhs_by_weight.keys()))

            if self.merge_strategy == MergeStrategy.WEIGHTS:
                for weight, rhs in rhs_by_weight.items():
                    B_rows.append(row)
                    F.append(rhs)
                    weights.append(weight)
            elif self.merge_strategy == MergeStrategy.DUPLICATES:
                current_rhs = set()
                for i in reversed(range(len(sorted_weights))):
                    weight = sorted_weights[i]
                    rhs = rhs_by_weight[weight]
                    n_duplicates = weight if i == 0 else weight - sorted_weights[i - 1]
                    current_rhs.update(rhs)
                    for _ in range(n_duplicates):
                        B_rows.append(row)
                        F.append(current_rhs.copy())
            elif self.merge_strategy == MergeStrategy.USE_STRICTEST:
                rhs = rhs_by_weight[sorted_weights[-1]]
                B_rows.append(row)
                F.append(rhs)
            elif self.merge_strategy == MergeStrategy.USE_LOOSEST:
                total_rhs = set()
                for rhs in rhs_by_weight.values():
                    total_rhs.update(rhs)
                B_rows.append(row)
                F.append(total_rhs)
            else:
                raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

        if self.equal_size_F_i:
            gcd = 0
            for F_i in F:
                gcd = math.gcd(gcd, len(F_i))
            new_B_rows = []
            new_F = []

            new_weights = []
            weights = weights if len(weights) > 0 else [None] * len(B_rows)

            assert len(B_rows) == len(F) == len(weights)
            for B_row, F_i, weight in zip(B_rows, F, weights):
                F_i_els = list(F_i)
                for i in range(0, len(F_i_els), gcd):
                    new_B_rows.append(B_row)
                    if weight is not None:
                        new_weights.append(weight)
                    new_F.append(set(F_i_els[i : i + gcd]))
            B_rows = new_B_rows
            F = new_F
            weights = new_weights

        self.B = Matrix(self.field, B_rows)
        self.F = F
        self.weights = weights

        if len(used_variables) != len(self.variables):
            unused_variables = []
            for var in self.variables:
                if var.id not in used_variables:
                    unused_variables.append(var)
            warnings.warn(
                f"Unused variables: {", ".join([var.name for var in unused_variables])}"
            )

        if self.B.nrows() <= self.B.ncols():
            warnings.warn(
                "Max-LINSAT instance should have more constraints than variables"
            )

    def compute_code(self):
        self.code = codes.from_parity_check_matrix(self.get_B().T)

    def get_B(self) -> Matrix:
        if self.B is None:
            self.compute_B_F_weights()
        return self.B

    def get_F(self) -> list[set[FiniteRingElement]]:
        if self.F is None:
            self.compute_B_F_weights()
        return self.F

    def get_weights(self) -> list[float]:
        if self.weights is None:
            self.compute_B_F_weights()
        if len(self.weights) == 0:
            return None
        return self.weights

    def get_code(self) -> AbstractLinearCode:
        if self.code is None:
            self.compute_code()
        return self.code

    def _compute_graph_minimum_distance(self):
        B = self.get_B()

        # singles: constraints with only 1 variable
        singles = []
        # double: constraints with exactly 2 variables
        doubles = []

        # if the instance contains only singles and doubles, this method returns the exact minimum distance
        # otherwise the result will be an upper bound on the minimum distance
        only_singles_and_doubles = True

        for row in B.rows():
            non_zero = [j for j, x in enumerate(row) if x != 0]
            if len(non_zero) == 1:
                singles.append(non_zero[0])
            elif len(non_zero) == 2:
                doubles.append((non_zero[0], non_zero[1]))
            else:
                only_singles_and_doubles = False

        graph = nx.Graph()
        graph.add_nodes_from(list(range(B.ncols())))
        for u, v in doubles:
            graph.add_edge(u, v)

        # girth: length of shortest cycle
        girth = nx.girth(graph)

        shortest_distance = math.inf
        for i, u in enumerate(singles):
            distances = nx.single_source_shortest_path_length(graph, u)
            for j in range(i + 1, len(singles)):
                v = singles[j]
                if v in distances and distances[v] < shortest_distance:
                    shortest_distance = distances[v]

        # a cycle or a path with two endpoints is a linear dependency (thus + 2)
        return min(girth, shortest_distance + 2), only_singles_and_doubles

    def _compute_minimum_distance_from_singles(self):
        B = self.get_B()
        singles = set()

        for row in B.rows():
            non_zero = [j for j, x in enumerate(row) if x != 0]
            if len(non_zero) == 1:
                singles.add(non_zero[0])

        smallest_combination = math.inf
        for row in B.rows():
            non_zero = [j for j, x in enumerate(row) if x != 0]
            if len(non_zero) == 1:
                continue
            if all(x in singles for x in non_zero):
                smallest_combination = min(smallest_combination, len(non_zero))
        return smallest_combination + 1

    def compute_minimum_distance(self, approximate=True):
        B = self.get_B()

        if B.T.right_kernel().dimension() == 0:
            return 0

        rows = list(B.rows())
        # check for duplicate rows, which automatically lead to minimum distance 2
        if len(set(rows)) < len(rows):
            return 2

        result_graph, is_exact = self._compute_graph_minimum_distance()
        if result_graph == 3 or is_exact:
            return result_graph

        result_singles = self._compute_minimum_distance_from_singles()
        if result_singles == 3:
            return result_singles

        if self.field.order() == 2:
            B = np.array(self.get_B().T).astype(np.int8)
            H = sp.sparse.csc_matrix(B)
            if approximate:
                return ldpc.code_util.estimate_code_distance(H)[0]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        return ldpc.code_util.compute_exact_code_distance(H)
                    except ValueError:
                        return 0
        else:
            code = self.get_code()
            if approximate:
                start = time.time()
                duration = 5

                smallest_weight = math.inf
                while time.time() < start + duration:
                    prob = random.random()
                    codeword = code.random_element(prob=prob)
                    weight = codeword.hamming_weight()
                    if weight == 0:
                        continue
                    smallest_weight = min(smallest_weight, weight)

                return min(smallest_weight, result_graph, result_singles)

            try:
                return code.minimum_distance("guava")
            except GAPError:
                return 0

    def get_minimum_distance(self) -> int:
        if self.minimum_distance is None:
            self.minimum_distance = self.compute_minimum_distance()
        return self.minimum_distance

    def __str__(self) -> str:
        return f"Max-LINSAT instance over {self.field} with {len(self.variables)} varaiables and {len(self.constraints)} constraints"


def MaxXorSat(
    default_decoder_constructor=NearestNeighborDecoder.constructor(),
    merge_strategy=MergeStrategy.DUPLICATES,
) -> MaxLinSat:
    return MaxLinSat(GF(2), default_decoder_constructor, merge_strategy)


class VertexColor(MaxLinSat):
    def __init__(
        self,
        n_colors: int,
        graph: nx.Graph,
        decoder_constructor=InformationSetDecoder.constructor(),
    ):
        super().__init__(GF(n_colors), decoder_constructor)

        variables = {u: self.new_var(f"x_{u}") for u in graph}

        for u, v in graph.edges:
            self.add_constraint(variables[u] != variables[v])


def MaxCut(graph: nx.Graph, decoder_constructor=BeliefPropagationDecoder.constructor()):
    return VertexColor(2, graph, decoder_constructor)


class BinaryPaintshop(MaxLinSat):
    @staticmethod
    def random_instances(n_cars: int, seed: int | None = None):
        cars = list(range(n_cars))
        jobs = cars + cars

        rng = random.Random(seed)
        rng.shuffle(jobs)
        return BinaryPaintshop(jobs)

    def __init__(self, jobs: list[int]):
        super().__init__(GF(2))
        variables = {}

        for value in jobs:
            if value not in variables:
                variables[value] = self.new_var(f"x_{value}")

        indexes = defaultdict(list)
        for i, v in enumerate(jobs):
            indexes[v].append(i)

        for idx in indexes.values():
            assert len(idx) == 2

        def get_color_for_index(i):
            value = jobs[i]
            if i == indexes[value][0]:
                return variables[value]
            if i == indexes[value][1]:
                return 1 - variables[value]
            raise ValueError("Unreachable")

        for i, (a, b) in enumerate(zip(jobs, jobs[1:])):
            color_a = get_color_for_index(i)
            color_b = get_color_for_index(i + 1)
            self.add_constraint(color_a == color_b)


def get_next_free_vertex(graph: nx.Graph) -> int:
    return max(u if isinstance(u, int) else -1 for u in graph) + 1


def prepare_hamiltonian_cycle_graph(graph: nx.Graph, seed: int) -> nx.Graph:
    p = next_prime(len(graph) - 1)
    if p == len(graph):
        return graph

    next_free = get_next_free_vertex(graph)
    new_graph = graph.copy()
    rng = random.Random(seed)
    while len(new_graph) < p:
        u, v = rng.choice(list(new_graph.edges))
        w = next_free
        new_graph.remove_edge(u, v)
        new_graph.add_edge(u, w)
        new_graph.add_edge(w, v)
        next_free += 1
    return new_graph


def add_hamiltonian_cycle_constraints(
    problem: MaxLinSat, graph: nx.Graph, first_vertex: int, random_seed: int
) -> dict[int, LinSatVar]:
    # x[u] = position of vertex u in the round trip
    x = {}

    for u in graph:
        # by symmetry, fix the first vertex to be at position 0
        if u == first_vertex:
            x[u] = 0
        else:
            x[u] = problem.new_var(f"x_{u}")

    vertices = list(x.keys())
    for i, u in enumerate(vertices):
        for v in vertices[i + 1 :]:
            # distinct vertices must appear at different positions
            problem.add_constraint(x[u] != x[v])

    for u, v in nx.non_edges(graph):
        # if u and v are not adjacent they should not directly follow each other in the round trip
        problem.add_constraint(x[u] - x[v] != {-1, +1})

    return x


class HamiltonianCycle(MaxLinSat):
    def __init__(
        self,
        graph: nx.Graph,
        decoder_constructor=BeliefPropagationDecoder.constructor(),
        random_seed=0,
    ):
        graph = prepare_hamiltonian_cycle_graph(graph, random_seed)
        super().__init__(
            GF(len(graph)), decoder_constructor, MergeStrategy.USE_STRICTEST
        )

        self.graph = graph

        self.first_vertex = next(iter(graph))

        self.vertex_to_variable = add_hamiltonian_cycle_constraints(
            self, graph, self.first_vertex, random_seed
        )

        # first vertex has no associated variable
        del self.vertex_to_variable[self.first_vertex]

    def solution_vector_to_path(self, solution_vector: vector):
        var_to_pos = {}
        for var, pos in zip(self.variables, solution_vector):
            var_to_pos[var.name] = int(pos)

        pos_to_vertex = {0: self.first_vertex}
        for u, var in self.vertex_to_variable.items():
            pos_to_vertex[var_to_pos[var.name]] = u

        path = [pos_to_vertex[i] for i in range(len(self.graph))]
        path.append(pos_to_vertex[0])
        return path


spin_to_binary = {
    +1: 0,
    -1: 1,
}


class UnweightedIsing(MaxLinSat):
    def __init__(
        self,
        J: np.ndarray,
        h: np.ndarray,
        default_decoder_constructor=BeliefPropagationDecoder.constructor(),
    ):
        super().__init__(GF(2), default_decoder_constructor)
        assert len(J) == len(J[0]) == len(h)

        n = len(h)

        self.J = J
        self.h = h

        elements = set()
        for row in J:
            for el in row:
                elements.add(float(el))
        for el in h:
            elements.add(float(el))

        allowed_elements = {0.0, 1.0, -1.0}
        if elements.union(allowed_elements) != allowed_elements:
            raise ValueError("Only {0, -1, +1} allowed in J and h")

        x = {i: self.new_var(f"x_{i}") for i in range(n)}

        for i in range(n):
            if h[i] == 0:
                continue
            self.add_constraint(x[i] != spin_to_binary[h[i]])

        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] == 0:
                    continue
                self.add_constraint(x[i] + x[j] != spin_to_binary[J[i, j]])


def random_subset(l: list, k: int):
    indices = np.random.choice(len(l), k, replace=False)
    return {l[i] for i in indices}


class OptimalPolynomialIntersection(AbstractMaxLinSat):
    @staticmethod
    def random(
        field: GF,
        degree: int,
        r: int,
        decoder_constructor=GeneralizedReedSolomonDecoder.constructor(),
    ):
        y_values = list(field)
        x_values = y_values

        points = {x: random_subset(y_values, r) for x in x_values}

        return OptimalPolynomialIntersection(field, degree, points, decoder_constructor)

    def __init__(
        self,
        field: GF,
        degree: int,
        points: dict[FiniteRingElement, set[FiniteRingElement]],
        decoder_constructor=NearestNeighborDecoder,
    ):
        assert degree + 1 < len(points)
        variables = [LinSatVar(field, i, f"a_{i}") for i in range(degree + 1)]
        super().__init__(field, variables, decoder_constructor)
        self.points = points
        self.F = []
        for x in sorted(points.keys()):
            self.F.append(self.points[x])
        self.code = GeneralizedReedSolomonCode(
            evaluation_points=sorted(points.keys()), dimension=degree + 1
        )
        self.dual_code = self.code.dual_code()

    def get_B(self) -> Matrix:
        return self.dual_code.parity_check_matrix().T

    def get_F(self) -> list[set[FiniteRingElement]]:
        return self.F

    def get_code(self) -> AbstractLinearCode:
        return self.dual_code

    def get_minimum_distance(self) -> int:
        return self.dual_code.minimum_distance()
