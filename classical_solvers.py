from __future__ import annotations

import itertools
import random

import numpy as np
from ortools.sat.python import cp_model
from sage.all import vector
from simanneal import Annealer


class AbstractSolver:
    def __init__(self, instance):
        self.instance = instance.to_max_linsat()
        self.solution = None

    def compute_solution(self, instance):
        raise NotImplementedError()

    def get_solution(self):
        if self.solution is None:
            self.solution = self.compute_solution(self.instance)
        return self.solution

    def get_solution_quality(self):
        solution = self.get_solution()
        return self.instance.evaluate_solution(solution)


class OrToolsSolver(AbstractSolver):
    def __init__(self, instance, time_limit=None):
        super().__init__(instance)
        self._is_optimal = None
        self.time_limit = time_limit

    def compute_solution(self, instance):
        if not instance.field.is_prime_field():
            raise ValueError("This method only supports prime fields")

        p = int(instance.field.order())
        B = instance.get_B()
        F = instance.get_F()
        weights = instance.get_weights()

        n = B.ncols()
        m = B.nrows()

        B = np.array(B)

        model = cp_model.CpModel()

        x = [model.NewIntVar(0, p - 1, f"x_{i}") for i in range(n)]

        satisfied_vars = []

        for i in range(m):
            for rhs in F[i]:
                rhs = int(rhs) % p
                linear_sum = sum(B[i, j] * x[j] for j in range(n) if B[i, j] != 0)
                mod_var = model.NewIntVar(0, p - 1, f"mod_var_{i}")
                satisfied = model.NewBoolVar(f"satisfied_{i}_{rhs}")
                model.AddModuloEquality(mod_var, linear_sum, p)
                model.Add(mod_var == rhs).OnlyEnforceIf(satisfied)
                model.Add(mod_var != rhs).OnlyEnforceIf(satisfied.Not())

                satisfied_vars.append(satisfied)

        if weights is None:
            model.Maximize(sum(satisfied_vars))
        else:
            model.Maximize(sum([v * w for v, w in zip(satisfied_vars, weights)]))

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 8
        if self.time_limit is not None:
            solver.parameters.max_time_in_seconds = self.time_limit
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self._is_optimal = status == cp_model.OPTIMAL
            return vector(
                instance.field, np.array([solver.Value(x[i]) for i in range(n)])
            )
        else:
            raise ValueError("Could not find a solution")

    def is_optimal(self):
        if self._is_optimal is None:
            self.get_solution()
        return self._is_optimal


class BruteForceSolver(AbstractSolver):
    def __init__(self, instance):
        super().__init__(instance)
        self.solution = None

    def compute_solution(self, instance):
        solutions = itertools.product(list(instance.field), repeat=instance.get_n())
        best_solution = None
        best_solution_value = None
        for solution in solutions:
            value = instance.evaluate_solution(solution)
            if best_solution_value is None or value > best_solution_value:
                best_solution_value = value
                best_solution = solution
        return best_solution


class MaxLinSatAnneal(Annealer):
    def __init__(self, instance, initial_state=None):
        self.instance = instance
        self.els = list(self.instance.field)
        if initial_state is None:
            initial_state = vector(
                self.instance.field,
                [random.choice(self.els) for _ in range(self.instance.get_n())],
            )
        self.state = initial_state

    def move(self):
        n = self.instance.get_n()
        move = vector(self.instance.field, [0] * n)
        index = random.choice(range(n))
        el = random.choice(self.els)
        move[index] = el
        self.state += move

    def energy(self):
        return self.instance.get_m() - self.instance.evaluate_solution(self.state)

    def update(self, *args):
        pass


class SimAnnealSolver(AbstractSolver):
    def __init__(
        self,
        instance,
        initial_state=None,
        Tmax=25000.0,
        Tmin=2.5,
        steps=10000,
    ):
        super().__init__(instance)
        self.initial_state = initial_state
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.steps = steps

    def compute_solution(self, instance):
        annealer = MaxLinSatAnneal(instance, self.initial_state)
        annealer.Tmax = self.Tmax
        annealer.Tmin = self.Tmin
        annealer.steps = self.steps

        return annealer.anneal()[0]


class PrangeSolver(AbstractSolver):
    def __init__(self, instance):
        super().__init__(instance)
        n = instance.get_n()
        m = instance.get_m()
        B = instance.get_B()

        if not (m >= n and B.rank() == n):
            raise ValueError(
                "To apply Prange's algorithm, there must exist at least as many constraints as variables and the constraint matrix must have full rank"
            )

    def get_expected_solution_quality(self):
        # We assume independence:
        # for each constraint, there is a n / m chance to be in the pivot elements
        # if it is in the pivot elements it is satisfied with probability 1
        # otherwise it is satisfied with probability F_i / q
        n = self.instance.get_n()
        m = self.instance.get_m()
        q = self.instance.field.order()
        return sum(
            n / m * 1 + (1 - n / m) * len(F_i) / q for F_i in self.instance.get_F()
        )

    def compute_solution(self, instance):
        expected_solution_quality = self.get_expected_solution_quality()
        for _ in range(1000):
            solution = self.compute_arbitrary_solution(instance)
            if instance.evaluate_solution(solution) >= expected_solution_quality:
                return solution
        raise ValueError("Could not find a solution of the expected quality")

    def compute_arbitrary_solution(self, instance):
        n = instance.get_n()
        m = instance.get_m()
        B = instance.get_B()
        F = instance.get_F()

        v = [random.choice(list(F_i)) for F_i in F]

        perm = list(range(m))
        random.shuffle(perm)
        shuffled_B = B[perm, :]
        pivots = shuffled_B.pivots()
        selected_rows = [perm[i] for i in pivots]

        selected_rows_set = set(selected_rows)
        B_subset = B[selected_rows, :]
        v_subset = vector(
            instance.field, [v_i for i, v_i in enumerate(v) if i in selected_rows]
        )

        x = B_subset.solve_right(v_subset)
        return x
