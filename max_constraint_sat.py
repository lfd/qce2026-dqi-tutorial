from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from enum import Enum
from fractions import Fraction

from sage.all import GF, is_prime, next_prime

from constraints import LinSatConstraint, LinSatTerm, LinSatTerms, LinSatVar
from decoders import DEFAULT_DECODER_CONSTRUCTOR
from max_lin_sat import MaxLinSat, MergeStrategy

type IntExpr = int | Fraction | IntVar | IntTerms


class Relation(Enum):
    EQUALS = 0
    DOES_NOT_EQUAL = 1
    LESS_THAN = 2
    GREATER_THAN = 3
    LESS_THAN_OR_EQUALS = 4
    GREATER_THAN_OR_EQUALS = 5

    def __str__(self) -> str:
        return {
            Relation.EQUALS: "=",
            Relation.DOES_NOT_EQUAL: "!=",
            Relation.LESS_THAN: "<",
            Relation.GREATER_THAN: ">",
            Relation.LESS_THAN_OR_EQUALS: "<=",
            Relation.GREATER_THAN_OR_EQUALS: ">=",
        }[self]


class IntConstraint:
    def __init__(
        self,
        terms: IntTerms,
        relation: Relation,
        mod: int | None = None,
        required_field_order: tuple[int | None, int] | None = None,
        weight: int = 1,
    ):
        if relation not in Relation:
            raise ValueError(f"Invalid relation {relation}")
        if not (mod is None or isinstance(mod, int)):
            raise ValueError(f"Only constraints modulo an integer supported: {mod}")
        self.terms = terms
        self.relation = relation
        self.mod = mod
        self.required_field_order = None
        if self.required_field_order is None:
            self.required_field_order = self.compute_required_field_order()
        self.weight = Fraction(weight)

    def with_mod(self, p: int) -> IntConstraint:
        if self.mod is not None:
            raise ValueError("Nested modulo in constraints is not allowed")
        if self.relation not in [Relation.EQUALS, Relation.DOES_NOT_EQUAL]:
            raise ValueError("Modulo constraints only support = and !=")
        return IntConstraint(self.terms, self.relation, p)

    def __mod__(self, p: int) -> IntConstraint:
        return self.with_mod(p)

    def with_weight(self, weight: Fraction) -> IntConstraint:
        return IntConstraint(
            self.terms, self.relation, self.mod, self.required_field_order, weight
        )

    def compute_required_field_order(self) -> tuple[int | None, int]:
        lcm = self.terms.compute_lcm()
        l, u = self.terms.get_lower_upper_bounds()
        lower_bound, upper_bound = int(l * lcm), int(u * lcm)

        safe_order = upper_bound - lower_bound + 1
        if self.relation in [
            Relation.GREATER_THAN,
            Relation.LESS_THAN,
            Relation.GREATER_THAN_OR_EQUALS,
            Relation.LESS_THAN_OR_EQUALS,
        ]:
            return None, safe_order
        elif self.relation in [Relation.EQUALS, Relation.DOES_NOT_EQUAL]:
            if self.mod is not None:
                return self.mod * lcm, safe_order
            else:
                return None, max(abs(lower_bound), abs(upper_bound)) + 1
        else:
            raise ValueError(f"Invalid constraint {self}")

    def __str__(self) -> str:
        parts = [f"{self.terms} {self.relation} 0"]
        if self.mod is not None:
            parts.append(f"(mod {self.mod})")
        if self.weight != 1:
            parts.append(f"[times {self.weight}]")
        return " ".join(parts)

    def get_degree(self) -> int:
        return self.terms.get_degree()

    def integerize(self) -> IntConstraint:
        lcm = self.terms.compute_lcm()
        return IntConstraint(
            self.terms.scale(lcm),
            self.relation,
            self.mod * lcm if self.mod is not None else self.mod,
            self.required_field_order,
            self.weight,
        )

    def get_degenerate_value(self) -> bool | None:
        l, u = self.terms.get_lower_upper_bounds()

        if self.mod is not None:
            return None

        if self.relation == Relation.EQUALS:
            if l == u == 0:
                return True
            if l > 0 or u < 0:
                return False
        elif self.relation == Relation.DOES_NOT_EQUAL:
            if l == u == 0:
                return False
            if l > 0 or u < 0:
                return True
        elif self.relation == Relation.GREATER_THAN:
            if l > 0:
                return True
            if u <= 0:
                return False
        elif self.relation == Relation.LESS_THAN:
            if u < 0:
                return True
            if l >= 0:
                return False
        elif self.relation == Relation.GREATER_THAN_OR_EQUALS:
            if l >= 0:
                return True
            if u < 0:
                return False
        elif self.relation == Relation.LESS_THAN_OR_EQUALS:
            if u <= 0:
                return True
            if l > 0:
                return False
        return None

    def to_lin_sat_constraint(
        self, field: GF, var_mapping: dict[int, LinSatVar]
    ) -> LinSatConstraint:
        if self.get_degree() > 1:
            raise ValueError(
                f"to_lin_sat_constraint does not support non-linear constraint {self}"
            )
        if self.terms.compute_lcm() != 1:
            raise ValueError(
                f"to_lin_constraint does not support non-integer coefficients: {self}"
            )

        p = field.order()

        degenerate_value = self.get_degenerate_value()

        if degenerate_value == True:
            return LinSatTerms.from_term_list(field, []) == 0
        elif degenerate_value == False:
            return LinSatTerms.from_term_list(field, []) == 1

        l, u = self.terms.get_lower_upper_bounds()
        l, u = int(l), int(u)

        term_list = []
        for term, coef in self.terms.terms.items():
            if len(term) == 0:
                term_list.append(LinSatTerm.constant(field, coef))
            else:
                assert len(term) == 1
                var_id = term[0][0]
                term_list.append(LinSatTerm(int(coef), var_mapping[var_id]))

        terms = LinSatTerms.from_term_list(field, term_list)

        if self.mod is None:
            if self.relation == Relation.EQUALS:
                return terms == 0
            elif self.relation == Relation.DOES_NOT_EQUAL:
                return terms != 0
            elif self.relation == Relation.GREATER_THAN:
                return terms == set(range(1, u + 1))
            elif self.relation == Relation.LESS_THAN:
                return terms == set(range(l, 0))
            elif self.relation == Relation.GREATER_THAN_OR_EQUALS:
                return terms == set(range(u + 1))
            elif self.relation == Relation.LESS_THAN_OR_EQUALS:
                return terms == set(range(l, 1))
            else:
                raise ValueError("Unreachable")
        else:
            rhs = set()
            assert self.relation in [Relation.EQUALS, Relation.DOES_NOT_EQUAL]
            invert = self.relation == Relation.DOES_NOT_EQUAL
            for value in range(l, u + 1):
                match = value % self.mod == 0
                if (match and not invert) or (not match and invert):
                    rhs.add(value)
            return terms == rhs


class IntVar:
    def __init__(
        self, var_id: int, name: str | None = None, lower: int = 0, upper: int = 1
    ):
        self.id = var_id
        self.name = name
        self.lower = lower
        self.upper = upper

    def is_binary(self) -> bool:
        return self.lower == 0 and self.upper == 1

    def __neg__(self) -> IntTerms:
        return -IntTerms.from_var(self)

    def __rmul__(self, coef: IntExpr) -> IntTerms:
        return coef * IntTerms.from_var(self)

    def __mul__(self, coef: IntExpr) -> IntTerms:
        return IntTerms.from_var(self) * coef

    def __add__(self, other: IntExpr) -> IntTerms:
        return IntTerms.from_var(self) + other

    def __sub__(self, other: IntExpr) -> IntTerms:
        return IntTerms.from_var(self) - other

    def __radd__(self, other: IntExpr) -> IntTerms:
        return other + IntTerms.from_var(self)

    def __rsub__(self, other: IntExpr):
        return other - IntTerms.from_var(self)

    def __eq__(self, other: IntExpr) -> IntTerms:
        return IntTerms.from_var(self) == other

    def __req__(self, other: IntExpr) -> IntConstraint:
        return other == IntTerms.from_var(self)

    def __ne__(self, other: IntExpr) -> IntConstraint:
        return IntTerms.from_var(self) != other

    def __rne__(self, other: IntExpr) -> IntConstraint:
        return other != IntTerms.from_var(self)

    def __lt__(self, other: IntExpr) -> IntConstraint:
        return IntTerms.from_var(self) < other

    def __gt__(self, other: IntExpr) -> IntConstraint:
        return IntTerms.from_var(self) > other

    def __le__(self, other: IntExpr) -> IntConstraint:
        return IntTerms.from_var(self) <= other

    def __ge__(self, other: IntExpr) -> IntConstraint:
        return IntTerms.from_var(self) >= other

    def __and__(self, other: IntExpr) -> IntTerms:
        return IntTerms.from_var(self) & other

    def __or__(self, other: IntExpr) -> IntTerms:
        return IntTerms.from_var(self) | other

    def __xor__(self, other: IntExpr) -> IntTerms:
        return IntTerms.from_var(self) ^ other

    def __invert__(self) -> IntTerms:
        return ~IntTerms.from_var(self)


TermKey = tuple[tuple[IntVar, int], ...]


class IntTerms:
    @staticmethod
    def from_var(var: IntVar) -> IntTerms:
        return IntTerms({var.id: var}, {((var.id, 1),): Fraction(1)})

    @staticmethod
    def from_scalar(coef: int | float | Fraction) -> IntTerms:
        return IntTerms({}, {(): Fraction(coef)})

    @staticmethod
    def from_any(other: IntTerms | int | Fraction | IntVar) -> IntTerms:
        if isinstance(other, IntTerms):
            return other
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, Fraction)
        ):
            return IntTerms.from_scalar(other)
        if isinstance(other, IntVar):
            return IntTerms.from_var(other)
        raise ValueError(f"Cannot convert {other} to IntTerms")

    @staticmethod
    def mul_terms(
        variables: dict[int, IntVar],
        a: TermKey,
        b: TermKey,
    ) -> TermKey:
        exps = {}
        for v, e in a + b:
            if v not in exps:
                exps[v] = 0
            exps[v] += e

        for v in exps:
            if variables[v].is_binary():
                exps[v] = min(1, exps[v])

        result = list(exps.items())
        result.sort(key=lambda x: x[0])
        return tuple(result)

    @staticmethod
    def merge_variables(
        a: dict[int, IntVar], b: dict[int, IntVar]
    ) -> dict[int, IntVar]:
        variables = a.copy()
        for var_id, var in b.items():
            if var_id in variables and variables[var_id] is not var:
                raise ValueError(
                    f"Variable mismatch for ID {var_id}: {variables[var_id]} != {var}"
                )
            variables[var_id] = var
        return variables

    def __init__(self, variables: dict[int, IntVar], terms: dict[TermKey, Fraction]):
        self.variables = variables
        self.terms = terms
        self.bounds = None

    def get_lower_upper_bounds(self) -> tuple[int, int]:
        if self.bounds is None:
            self.bounds = self.compute_lower_upper_bounds()
        return self.bounds

    def compute_lower_upper_bounds(self) -> tuple[int, int]:
        lower_bound = 0
        upper_bound = 0
        for variables, coef in self.terms.items():
            term_min = coef
            term_max = coef
            for var, e in variables:
                l = self.variables[var].lower ** e
                u = self.variables[var].upper ** e
                new_term_min = min(
                    term_min * l, term_min * u, term_max * l, term_max * u
                )
                new_term_max = max(
                    term_min * l, term_min * u, term_max * l, term_max * u
                )
                term_min = new_term_min
                term_max = new_term_max
            lower_bound += term_min
            upper_bound += term_max
        return lower_bound, upper_bound

    def compute_lcm(self) -> int:
        return math.lcm(*[coef.denominator for coef in self.terms.values()])

    def add(self, other: IntExpr) -> IntTerms:
        other = IntTerms.from_any(other)
        variables = IntTerms.merge_variables(self.variables, other.variables)
        terms = self.terms.copy()
        for term, coef in other.terms.items():
            if term not in terms:
                terms[term] = 0
            terms[term] += coef

        for term in list(terms.keys()):
            if terms[term] == 0:
                del terms[term]

        return IntTerms(variables, terms)

    def scale(self, scalar: Fraction | int) -> IntTerms:
        if not (isinstance(scalar, int) or isinstance(scalar, Fraction)):
            raise ValueError(f'{scalar} must be of type "int" or "Fraction"')
        if scalar == 0:
            return IntTerms({}, {})
        terms = {term: coef * scalar for term, coef in self.terms.items()}
        return IntTerms(self.variables, terms)

    def scalar_div(self, scalar: Fraction | int) -> IntTerms:
        if not (isinstance(scalar, int) or isinstance(scalar, Fraction)):
            raise ValueError(f'{scalar} must be of type "int" or "Fraction"')
        if scalar == 0:
            return IntTerms({}, {})
        terms = {term: coef / scalar for term, coef in self.terms.items()}
        return IntTerms(self.variables, terms)

    def mul(self, other: IntExpr) -> IntTerms:
        other = IntTerms.from_any(other)
        variables = IntTerms.merge_variables(self.variables, other.variables)

        terms = {}
        for term1, coef1 in self.terms.items():
            for term2, coef2 in other.terms.items():
                new_term = IntTerms.mul_terms(variables, term1, term2)
                if new_term not in terms:
                    terms[new_term] = 0
                terms[new_term] += coef1 * coef2

        return IntTerms(variables, terms)

    def to_constraint(
        self, other: int | IntVar | IntTerms, relation: Relation
    ) -> IntConstraint:
        other = IntTerms.from_any(other)
        new_terms = self - other
        return IntConstraint(new_terms, relation)

    def is_binary(self) -> bool:
        for term in self.terms:
            for v, _ in term:
                if not self.variables[v].is_binary():
                    return False
        return True

    def get_degree(self) -> int:
        return max((sum(t[1] for t in term) for term in self.terms), default=0)

    def to_linearized_constraint(self) -> list[IntConstraint]:
        if self.is_binary():
            return self.to_ising()
        if self.get_degree() <= 1:
            return self.to_non_binary_linearized_constraint()
        raise ValueError(
            f"Non-binary higher-order objective cannot be linearized: {self}"
        )

    def to_non_binary_linearized_constraint(self) -> list[IntConstraint]:
        constraints = []
        constant_offset = 0

        for term, c in self.terms.items():
            if len(term) == 0:
                constant_offset += c
                continue
            if len(term) > 1 or term[0][1] != 1:
                raise ValueError(
                    f"Non-binary higher-order objective cannot be linearized: {self}"
                )
            variable = self.variables[term[0][0]]

            if c > 0:
                constant_offset += variable.lower * c
            else:
                constant_offset += variable.upper * c

            for num in range(variable.lower, variable.upper + 1):
                if c > 0:
                    weight = c * (num - variable.lower)
                else:
                    weight = -c * (variable.upper - num)
                if weight != 0:
                    constraints.append((variable == num).with_weight(weight))

        constraints.append(
            IntConstraint(IntTerms({}, {}), Relation.EQUALS, weight=constant_offset)
        )

        return constraints

    def to_ising(self) -> list[IntConstraint]:
        if not self.is_binary():
            raise ValueError(f"to_ising only supports binary variables: {self}")

        ising = {}

        offset = 0

        for term, c in self.terms.items():
            variables = list((v for v, _ in term))
            factor = Fraction(c, 2 ** len(variables))
            for k in range(len(variables) + 1):
                sign = 1 if k % 2 == 0 else -1
                coef = factor * sign
                for comb in itertools.combinations(variables, k):
                    if comb not in ising:
                        ising[comb] = 0
                    ising[comb] += coef

        if () in ising:
            ising[()] -= sum(
                (abs(c) if c != 0 and term != () else 0 for term, c in ising.items())
            )

        constraints = []
        for term, c in ising.items():
            if c == 0:
                continue
            if term == ():
                constraints.append(
                    IntConstraint(IntTerms({}, {}), Relation.EQUALS, weight=c)
                )
                continue

            variables = {i: self.variables[i] for i in term}
            terms = {((i, 1),): Fraction(1) for i in term}
            if c < 0:
                terms[()] = 1

            int_terms = IntTerms(variables, terms)
            constraints.append(
                IntConstraint(int_terms, Relation.EQUALS, mod=2, weight=2 * abs(c))
            )
        return constraints

    def reduce_degree(
        self,
        get_temp_var: Callable[[tuple[int, ...]], IntVar],
        max_degree: int,
        max_aux_degree: int,
    ) -> IntTerms:
        if not self.is_binary():
            raise ValueError("Degree reduction only supported for binary variables")

        new_terms = {}
        new_vars = {}

        temp_vars = {}
        for term, coef in self.terms.items():
            var_ids = [v for v, _ in term]
            while len(var_ids) > max_degree:
                new_var_ids = []
                for chunk in itertools.batched(var_ids, max_aux_degree):
                    if len(chunk) == 1:
                        new_var_ids.append(chunk[0])
                    else:
                        chunk_id = tuple(chunk)
                        temp_var = get_temp_var(chunk_id)
                        temp_vars[temp_var.id] = temp_var
                        new_var_ids.append(temp_var.id)
                var_ids = new_var_ids

            for var_id in var_ids:
                if var_id in temp_vars:
                    new_vars[var_id] = temp_vars[var_id]
                elif var_id in self.variables:
                    new_vars[var_id] = self.variables[var_id]
                else:
                    raise ValueError("unreachable")

            term_id = tuple([(var_id, 1) for var_id in var_ids])

            new_terms[term_id] = coef

        result = IntTerms(new_vars, new_terms)
        return result

    def linearize(
        self, get_temp_var: Callable[[tuple[int, ...]], IntVar], max_degree: int
    ) -> IntTerms:
        return self.reduce_degree(get_temp_var, max_degree=1, max_aux_degree=max_degree)

    def __mul__(self, coef: IntExpr) -> IntTerms:
        if isinstance(coef, int):
            return self.scale(coef)
        return self.mul(coef)

    def __truediv__(self, coef: IntExpr) -> IntTerms:
        return self.scalar_div(coef)

    def __rmul__(self, coef: IntExpr) -> IntTerms:
        return self * coef

    def __neg__(self) -> IntTerms:
        return self * (-1)

    def __add__(self, other: IntExpr) -> IntTerms:
        return self.add(other)

    def __radd__(self, other: IntExpr) -> IntTerms:
        return self + other

    def __sub__(self, other: IntExpr) -> IntTerms:
        return self + (-other)

    def __rsub__(self, other: IntExpr) -> IntTerms:
        return -self + other

    def __and__(self, other: IntExpr) -> IntTerms:
        other = IntTerms.from_any(other)
        return self * other

    def __or__(self, other: IntExpr) -> IntTerms:
        other = IntTerms.from_any(other)
        return self + other - self * other

    def __xor__(self, other: IntExpr) -> IntTerms:
        other = IntTerms.from_any(other)
        return self + other - 2 * self * other

    def __invert__(self) -> IntTerms:
        return 1 - self

    def __rand__(self, other: IntExpr) -> IntTerms:
        return self & other

    def __ror__(self, other: IntExpr) -> IntTerms:
        return self | other

    def __rxor__(self, other: IntExpr) -> IntTerms:
        return self ^ other

    def __eq__(self, other: IntExpr) -> IntConstraint:
        return self.to_constraint(other, Relation.EQUALS)

    def __ne__(self, other: IntExpr) -> IntConstraint:
        return self.to_constraint(other, Relation.DOES_NOT_EQUAL)

    def __lt__(self, other: IntExpr) -> IntConstraint:
        return self.to_constraint(other, Relation.LESS_THAN)

    def __gt__(self, other: IntExpr) -> IntConstraint:
        return self.to_constraint(other, Relation.GREATER_THAN)

    def __le__(self, other: IntExpr) -> IntConstraint:
        return self.to_constraint(other, Relation.LESS_THAN_OR_EQUALS)

    def __ge__(self, other: IntExpr) -> IntConstraint:
        return self.to_constraint(other, Relation.GREATER_THAN_OR_EQUALS)

    def __req__(self, other: IntExpr) -> IntConstraint:
        return self == other

    def __rne__(self, other: IntExpr) -> IntConstraint:
        return self != other

    def __rlt__(self, other: IntExpr) -> IntConstraint:
        return self > other

    def __rgt__(self, other: IntExpr) -> IntConstraint:
        return self < other

    def __rle__(self, other: IntExpr) -> IntConstraint:
        return self >= other

    def __rge__(self, other: IntExpr) -> IntConstraint:
        return self <= other

    def __str__(self) -> str:
        output = []
        for t, c in self.terms.items():
            if len(output) == 0:
                if c != 1 or len(t) == 0:
                    output.append(str(c))
            elif c < 0:
                output.append("-")
                if c != -1 or len(t) == 0:
                    output.append(str(-c))
            else:
                output.append("+")
                if c != 1 or len(t) == 0:
                    output.append(str(c))
            for v, e in t:
                if e == 1:
                    output.append(self.variables[v].name)
                else:
                    output.append(f"{self.variables[v].name}^{e}")
        if output == []:
            return "0"
        else:
            return " ".join(output)


class MaxConstraintSat:
    def __init__(self):
        self.variables: list[IntVar] = []
        self.constraints: list[IntConstraint] = []
        self.objectives: list[IntTerms] = []
        self.equality_constraints: list[IntTerms] = []
        self.linearized_objectives: list[IntConstraint] | None = None
        self.linearized_equality_constraints: list[IntConstraint] | None = None

    def new_var(self, name: str, lower: int, upper: int) -> IntVar:
        var = IntVar(len(self.variables), name, lower, upper)
        self.variables.append(var)
        return var

    def new_binary_var(self, name: str) -> IntVar:
        return self.new_var(name, 0, 1)

    def add_constraint(
        self,
        constraint: IntConstraint,
        mod: int | None = None,
        weight: Fraction | None = None,
    ):
        if mod is not None:
            constraint = constraint.with_mod(mod)
        if weight is not None:
            constraint = constraint.with_weight(weight)
        self.constraints.append(constraint)

    def add_objective(
        self, terms: IntTerms, minimize: bool = False, weight: Fraction | int = 1
    ):
        factor = weight
        if minimize:
            factor *= -1
        terms = terms * factor

        self.objectives.append(terms)

    def add_boolean_constraint(self, terms: IntTerms, weight: Fraction | int = 1):
        self.add_objective(terms, weight=weight)

    def add_boolean_equality(
        self,
        terms1: IntTerms,
        terms2: IntTerms,
        invert: bool = False,
        weight: Fraction | int = 1,
    ):
        difference = terms1 - terms2
        if invert:
            constraint = difference * difference
        else:
            constraint = 1 - difference * difference
        self.objectives.append(constraint * weight)

    def get_linearized_objectives(self) -> list[IntConstraint]:
        if self.linearized_objectives is None:
            self.linearized_objectives = []
            for constraint in self.objectives:
                self.linearized_objectives.extend(constraint.to_linearized_constraint())
        return self.linearized_objectives

    def get_linearized_equality_constraints(self) -> list[IntConstraint]:
        if self.linearized_equality_constraints is None:
            self.linearized_equality_constraints = []
            for constraint in self.equality_constraints:
                self.linearized_equality_constraints.extend(
                    constraint.to_linearized_constraint()
                )
        return self.linearized_equality_constraints

    def compute_field_order(self) -> int:
        strict_orders = set()
        loose_orders = set()
        loose_orders_without_strict = set()

        all_constraints = [
            *self.constraints,
            *self.get_linearized_objectives(),
            *self.get_linearized_equality_constraints(),
        ]

        for constraint in all_constraints:
            strict, loose = constraint.compute_required_field_order()
            if strict is not None:
                strict_orders.add(strict)
            else:
                loose_orders_without_strict.add(loose)
            loose_orders.add(loose)

        max_loose_order = max(loose_orders)

        if len(strict_orders) == 1:
            strict_order = next(iter(strict_orders))
            if is_prime(strict_order) and (
                len(loose_orders_without_strict) == 0
                or max(loose_orders_without_strict) <= strict_order
            ):
                return strict_order
        return next_prime(max_loose_order - 1)

    def get_constraint_degree(self) -> int:
        return max((c.get_degree() for c in self.constraints), default=0)

    def get_objective_degree(self) -> int:
        return max((o.get_degree() for o in self.objectives), default=0)

    def compute_weight_lcm(self) -> int:
        return math.lcm(
            *[
                c.weight.denominator
                for cs in [
                    self.constraints,
                    self.get_linearized_objectives(),
                    self.get_linearized_equality_constraints(),
                ]
                for c in cs
            ]
        )

    @staticmethod
    def create_equality_constraint(variables, term_key: TermKey, variable: IntVar):
        product = IntTerms.from_scalar(1)
        for var_id in term_key:
            product = product * variables[var_id]
        difference = product - variable
        return difference * difference

    def reduce_constraint_degrees(self, max_degree: int) -> MaxConstraintSat:
        temp_vars = {}
        problem = MaxConstraintSat()
        problem.variables = self.variables.copy()
        problem.equality_constraints = self.equality_constraints.copy()

        def get_temp_var(key):
            if key not in temp_vars:
                temp_var = problem.new_binary_var(f"temp_{len(temp_vars)}")
                temp_vars[key] = temp_var
                equality_constraint = MaxConstraintSat.create_equality_constraint(
                    problem.variables, key, temp_var
                )
                problem.equality_constraints.append(equality_constraint)
            return temp_vars[key]

        for objective in self.objectives:
            if objective.get_degree() <= max_degree:
                problem.add_objective(objective)
            else:
                new_objective = objective.reduce_degree(
                    get_temp_var, max_degree=max_degree, max_aux_degree=max_degree
                )
                problem.add_objective(new_objective)

        for constraint in self.constraints:
            if constraint.get_degree() <= 1:
                problem.add_constraint(constraint)
            else:
                problem.add_constraint(
                    IntConstraint(
                        constraint.terms.linearize(get_temp_var, max_degree),
                        constraint.relation,
                        constraint.mod,
                        constraint.required_field_order,
                        constraint.weight,
                    )
                )
        return problem

    def to_max_linsat(
        self,
        max_degree: int = 3,
        var_range_constraint_factor: int = 2,
        equality_constraint_factor: int = 2,
        default_decoder_constructor=DEFAULT_DECODER_CONSTRUCTOR,
        merge_strategy: MergeStrategy = MergeStrategy.DUPLICATES,
        equal_size_F_i: bool = True,
    ) -> MaxLinSat:
        if (
            len(self.constraints)
            + len(self.equality_constraints)
            + len(self.objectives)
            == 0
        ):
            raise ValueError("Cannot convert empty MaxConstraintSat instance")
        if self.get_constraint_degree() > 1 or self.get_objective_degree() > max_degree:
            return self.reduce_constraint_degrees(max_degree).to_max_linsat(
                max_degree=max_degree,
                var_range_constraint_factor=var_range_constraint_factor,
                equality_constraint_factor=equality_constraint_factor,
                default_decoder_constructor=default_decoder_constructor,
                merge_strategy=merge_strategy,
                equal_size_F_i=equal_size_F_i,
            )
        field = GF(self.compute_field_order())
        max_lin_sat = MaxLinSat(
            field=field,
            default_decoder_constructor=default_decoder_constructor,
            merge_strategy=merge_strategy,
            equal_size_F_i=equal_size_F_i,
        )
        var_map = {}

        lcm = self.compute_weight_lcm()
        total_weight = (
            len(self.objectives) + sum(c.weight for c in self.constraints) * lcm
        )

        var_range_weight = total_weight * var_range_constraint_factor
        equality_constraint_weight = total_weight * equality_constraint_factor

        for variable in self.variables:
            lin_sat_var = max_lin_sat.new_var(variable.name)
            var_map[variable.id] = lin_sat_var
            max_lin_sat.add_constraint(
                lin_sat_var == set(range(variable.lower, variable.upper + 1)),
                weight=var_range_weight,
                disable_warnings=True,
            )

        for constraint in self.get_linearized_equality_constraints():
            linsat_constraint = constraint.integerize().to_lin_sat_constraint(
                field, var_map
            )
            max_lin_sat.add_constraint(
                linsat_constraint,
                weight=equality_constraint_weight,
            )

        for constraint in [
            *self.constraints,
            *self.get_linearized_objectives(),
        ]:
            if constraint.weight < 0:
                # skip degenerate constraint
                if constraint.get_degenerate_value() is not None:
                    continue
                raise ValueError(
                    f"Cannot convert negative-weight constraint to Max-LINSAT: {constraint}"
                )
            max_lin_sat.add_constraint(
                constraint.integerize().to_lin_sat_constraint(field, var_map),
                weight=constraint.weight * lcm,
            )
        return max_lin_sat
