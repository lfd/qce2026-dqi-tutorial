from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import *

import numpy as np
from sage.all import GF, Matrix
from sage.arith.misc import is_prime
from sage.rings.finite_rings.element_base import FiniteRingElement


def is_valid_first_permutation_key(key: tuple[int, ...]) -> bool:
    for el in key:
        if el == 0:
            continue
        return el == 1
        if el == 1:
            return True
        return False


def permutation_orbits(p: int, k: int) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    orbits = defaultdict(list)
    for t in product(range(p), repeat=k):
        key = tuple(sorted(t))
        orbits[key].append(t)
    return dict(orbits)


def powerset(iterable: Iterable) -> Iterable:
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def count(p: int, B: np.ndarray, v: np.ndarray, x: np.ndarray) -> int:
    x = np.array(x).reshape(len(B[0]), 1)

    result = (B @ x) % p

    tally = 0

    for y_i, v_i in zip(result.reshape(len(B)), v):
        if y_i in v_i:
            tally += 1

    return tally


def rate_solution(p: int, B: np.ndarray) -> tuple[int, int]:
    m = B.shape[0]

    n_dependent = math.inf
    for k in range(1, m + 1):
        for indexes in combinations(range(m), k):
            for values in product(range(1, p), repeat=k):
                y = np.zeros((1, m))
                for v, i in zip(values, indexes):
                    y[0, i] = v
                if ((y @ B) % p == 0).all():
                    n_dependent = k
                    break
            if n_dependent == k:
                break
        if n_dependent != math.inf:
            break

    return (n_dependent, -m)


def is_valid_temp_B(temp_B: np.ndarray) -> bool:
    for constraint in temp_B:
        for s in constraint:
            if sum(c for c in s) == 0:
                continue
            if is_valid_first_permutation_key(s):
                break
            else:
                return False
    return True


def find_gadget(
    p: int,
    k: int,
    results: list[bool],
    symmetry_sets: list[list[int]] | None = None,
    strict_yes: bool = True,
    strict_no: bool = True,
    fixed_r: bool = True,
    n_aux_vars: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    k += n_aux_vars
    values = list(range(p))

    assert p ** (k - n_aux_vars) == len(results)

    if symmetry_sets is None:
        symmetry_sets = []

    all_values = set(range(k))
    for s in symmetry_sets:
        for v in s:
            assert v in all_values
            all_values.remove(v)

    for v in all_values:
        symmetry_sets.append([v])

    n = p**k

    best_solutions = []
    best_solution_rate = None

    n = p ** len(symmetry_sets)

    orbits = [permutation_orbits(p, len(s)) for s in symmetry_sets]

    for temp_B in powerset(
        c
        for c in product(*[orbit.keys() for orbit in orbits])
        if sum(v for s in c for v in s) > 0
    ):
        if len(temp_B) == 0:
            continue
        if not is_valid_temp_B(temp_B):
            continue

        if fixed_r:
            r_values = ([r] * len(temp_B) for r in range(1, p))
        else:
            r_values = product(range(1, p), repeat=len(temp_B))
        for rs in r_values:
            for temp_v in product(*[combinations(range(p), r) for r in rs]):
                B = []
                v = []
                for temp_b_i, v_i in zip(temp_B, temp_v):
                    for perm in product(
                        *[
                            orbit[temp_b_i_part]
                            for orbit, temp_b_i_part in zip(orbits, temp_b_i)
                        ]
                    ):
                        v.append(v_i)
                        b_i = [0] * k
                        for coefs, s in zip(perm, symmetry_sets):
                            for c, index in zip(coefs, s):
                                b_i[index] = c
                        B.append(b_i)

                B = np.array(B)

                success = True
                yes_minimum = None
                yes_maximum = None

                no_maximum = None
                no_minimum = None

                for x, yes in zip(product(values, repeat=k - n_aux_vars), results):
                    if n_aux_vars == 0:
                        c = count(p, B, v, x)
                    else:
                        c = max(
                            count(p, B, v, x + x2)
                            for x2 in product(values, repeat=n_aux_vars)
                        )

                    if yes:
                        if yes_minimum is None:
                            yes_minimum = c
                            yes_maximum = c
                        else:
                            yes_maximum = max(c, yes_maximum)
                            yes_minimum = min(c, yes_minimum)
                    else:
                        if no_minimum is None:
                            no_minimum = c
                            no_maximum = c
                        else:
                            no_maximum = max(c, no_maximum)
                            no_minimum = min(c, no_minimum)

                    if (
                        (strict_yes and yes_maximum != yes_minimum)
                        or (strict_no and no_maximum != no_minimum)
                        or (
                            no_maximum is not None
                            and yes_minimum is not None
                            and no_maximum >= yes_minimum
                        )
                    ):
                        success = False
                        break

                if success:
                    rate = rate_solution(p, B)
                    if best_solution_rate is None or rate > best_solution_rate:
                        best_solutions = [(B, v)]
                        best_solution_rate = rate
                    elif rate == best_solution_rate:
                        best_solutions.append((B, v))
    return best_solutions


class Gadget:
    @staticmethod
    def deserialize(d):
        field = GF(d["p"])
        return Gadget(
            d["name"],
            d["p"],
            Matrix(field, d["B"]),
            [{field(el) for el in F_i} for F_i in d["F"]],
            d["truth_table"],
            d["strict_yes"],
            d["strict_no"],
            d["fixed_r"],
            d["n_aux_vars"],
        )

    def __init__(
        self,
        name: str,
        p: int,
        B: Matrix,
        F: list[set[FiniteRingElement]],
        truth_table: list[bool],
        strict_yes: bool,
        strict_no: bool,
        fixed_r: bool,
        n_aux_vars: int,
    ):
        self.p = p
        assert is_prime(p)

        self.name = name
        self.field = GF(p)
        self.B = Matrix(self.field, B)
        self.n = self.B.ncols()
        self.m = self.B.nrows()
        self.F = [{self.field(f) for f in F_i} for F_i in F]
        self.truth_table = truth_table
        self.strict_yes = strict_yes
        self.strict_no = strict_no
        self.fixed_r = fixed_r
        self.n_aux_vars = n_aux_vars

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "p": self.p,
            "B": [[str(e) for e in row] for row in self.B.rows()],
            "F": [[str(el) for el in F_i] for F_i in self.F],
            "truth_table": self.truth_table,
            "strict_yes": self.strict_yes,
            "strict_no": self.strict_no,
            "fixed_r": self.fixed_r,
            "n_aux_vars": self.n_aux_vars,
        }

    def evaluate(self, x: np.ndarray) -> int:
        x = np.array(x)
        y = (self.B @ x) % self.p

        return sum(1 if y_i in F_i else 0 for y_i, F_i in zip(y, self.F))


class GadgetLibrary:
    def __init__(self, path: str):
        self.path = path
        self.gadgets = []
        self.gadgets_by_name = {}
        self.load()

    def get_gadget(self, name: str) -> Gadget:
        if name in self.gadgets_by_name:
            return self.gadgets_by_name[name]
        else:
            return None

    def add_gadget(self, gadget: Gadget):
        if gadget.name in self.gadgets_by_name:
            warnings.warn(f"Overriding existing gadget {gadget.name}")

        self.gadgets.append(gadget)
        self.gadgets_by_name[gadget.name] = gadget
        self.store()

    def add_gadget_from_spec(self, gadget_spec: GadgetSpec) -> Gadget:
        old_gadget = self.get_gadget(gadget_spec.name)
        if old_gadget is not None:
            return old_gadget

        gadget = gadget_spec.to_gadget()

        self.add_gadget(gadget)
        return gadget

    def load(self):
        with open(self.path) as file:
            data = json.load(file)
        self.gadgets = [Gadget.deserialize(d) for d in data]
        self.gadgets_by_name = {g.name: g for g in self.gadgets}

    def store(self):
        data = [g.serialize() for g in self.gadgets]
        with open(self.path, "w") as file:
            json.dump(data, file)


class GadgetSpec:
    def __init__(
        self,
        name: str,
        p: int,
        n: int,
        fn: Callable[[tuple], bool],
        symmetry_sets: list[list[int]] | None = None,
        strict_yes: bool = True,
        strict_no: bool = True,
        fixed_r: bool = True,
        n_aux_vars: int = 0,
    ):
        if symmetry_sets is None:
            symmetry_sets = []
        self.name = name
        self.p = p
        self.n = n
        self.fn = fn
        self.symmetry_sets = symmetry_sets
        self.strict_yes = strict_yes
        self.strict_no = strict_no
        self.fixed_r = fixed_r
        self.n_aux_vars = n_aux_vars

    def compute_truth_table(self) -> list[bool]:
        return [self.fn(*x) for x in product(range(self.p), repeat=self.n)]

    def to_gadget(self) -> Gadget:
        truth_table = self.compute_truth_table()

        gadgets = find_gadget(
            self.p,
            self.n,
            self.compute_truth_table(),
            self.symmetry_sets,
            self.strict_yes,
            self.strict_no,
            self.fixed_r,
            self.n_aux_vars,
        )
        B, F = gadgets[0]

        gadget = Gadget(
            self.name,
            self.p,
            B,
            F,
            truth_table,
            self.strict_yes,
            self.strict_no,
            self.fixed_r,
            self.n_aux_vars,
        )
        return gadget


library = GadgetLibrary("gadget_libaray.json")

gadget_specs = []

and_fn = lambda *args: sum(args) == len(args)
or_fn = lambda *args: sum(args) >= 1

for k in range(2, 5 + 1):
    symmetry_sets = [list(range(k))]
    gadget_specs.append(GadgetSpec(f"and{k}", 2, k, and_fn, symmetry_sets))
    gadget_specs.append(GadgetSpec(f"or{k}", 2, k, or_fn, symmetry_sets))

for spec in gadget_specs:
    library.add_gadget_from_spec(spec)

and2 = library.gadgets_by_name["and2"]
or2 = library.gadgets_by_name["or2"]

if __name__ == "__main__":
    gadget = GadgetSpec(
        "asdfasd",
        2,
        3,
        lambda a, b, c: (a + b >= 1) == (c == 1),
        symmetry_sets=[[0, 1], [3, 4]],
        strict_no=False,
        n_aux_vars=2,
    ).to_gadget()
    print(gadget.B)
    print(gadget.F)
    for x in product(range(gadget.p), repeat=gadget.n):
        print(x, gadget.evaluate(x))

    B = []
    F = []
    truth_table = []
    p = 3
    for a in range(p):
        for b in range(p):
            c = (a * b) % p
            B.append([a, b, -1])
            F.append({c})
            for _c in range(p):
                truth_table.append(c == _c)

    gadget = Gadget("test", p, np.array(B), F, truth_table, True, True, True)
    for i, x in enumerate(product(range(p), repeat=3)):
        print(x, gadget.evaluate(x), truth_table[i])
    print()
