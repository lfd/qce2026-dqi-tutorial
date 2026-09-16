from __future__ import annotations

import warnings
from collections.abc import Callable

from sage.all import GF
from sage.rings.finite_rings.element_base import FiniteRingElement

type LinSatExpr = int | FiniteRingElement | LinSatVar | LinSatTerms


class LinSatVar:
    def __init__(self, field: GF, id: int, name: str | None = None):
        self.field = field
        self.id = id
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __neg__(self) -> LinSatTerms:
        return -LinSatTerms.from_var(self)

    def __rmul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        return coef * LinSatTerms.from_var(self)

    def __mul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        return LinSatTerms.from_var(self) * coef

    def __add__(self, other: LinSatExpr) -> LinSatTerms:
        return LinSatTerms.from_var(self) + other

    def __sub__(self, other: LinSatExpr) -> LinSatTerms:
        return LinSatTerms.from_var(self) - other

    def __radd__(self, other: LinSatExpr) -> LinSatTerms:
        return other + LinSatTerms.from_var(self)

    def __rsub__(self, other: LinSatExpr) -> LinSatTerms:
        return other - LinSatTerms.from_var(self)

    def __eq__(self, other: LinSatExpr) -> LinSatTerms:
        return LinSatTerms.from_var(self) == other

    def __req__(self, other: LinSatExpr) -> LinSatConstraint:
        return other == LinSatTerms.from_var(self)

    def __ne__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) != other

    def __rne__(self, other: LinSatExpr) -> LinSatConstraint:
        return other != LinSatTerms.from_var(self)

    def __lt__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) < other

    def __gt__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) > other

    def __le__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) <= other

    def __ge__(self, other: LinSatExpr) -> LinSatConstraint:
        return LinSatTerms.from_var(self) >= other


class LinSatTerm:
    @staticmethod
    def constant(field: GF, coef: FiniteRingElement) -> LinSatTerm:
        var = LinSatVar(field, None, "1")
        return LinSatTerm(coef, var)

    def __init__(self, coef: FiniteRingElement, var: LinSatVar):
        self.field = var.field
        self.coef = self.field(coef)
        self.var = var

    def __str__(self) -> str:
        if self.var.id is None:
            return str(self.coef)
        return f"{self.coef} {self.var}"


class LinSatTerms:
    @staticmethod
    def from_var(var: LinSatVar) -> LinSatTerms:
        return LinSatTerms.from_term_list(var.field, [LinSatTerm(1, var)])

    @staticmethod
    def from_scalar(field: GF, coef: FiniteRingElement) -> LinSatTerms:
        return LinSatTerms.from_term_list(field, [LinSatTerm.constant(field, coef)])

    @staticmethod
    def from_any(field: GF, other: LinSatExpr) -> LinSatTerms:
        if isinstance(other, LinSatTerms):
            assert other.field == field
            return other
        if isinstance(other, int) or other in field:
            return LinSatTerms.from_scalar(field, other)
        if isinstance(other, LinSatVar):
            assert field == other.field
            return LinSatTerms.from_var(other)
        raise ValueError(f"Cannot convert {other} to Terms")

    @staticmethod
    def from_term_list(field: GF, term_list: list[LinSatTerm]) -> LinSatTerms:
        return LinSatTerms(field, {}).add_term_list(term_list)

    def __init__(self, field: GF, terms: dict[int, LinSatTerm]):
        self.field = field
        self.terms = terms

    def add_term_list(self, term_list: list[LinSatTerm]) -> LinSatTerms:
        terms = self.terms.copy()
        for term in term_list:
            vid = term.var.id
            if vid not in terms:
                terms[vid] = term
            else:
                original = terms[vid]
                terms[vid] = LinSatTerm(original.coef + term.coef, term.var)

        for vid in list(terms.keys()):
            if terms[vid].coef == self.field(0):
                del terms[vid]

        return LinSatTerms(self.field, terms)

    def __mul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        coef = self.field(coef)

        if coef == self.field(0):
            return LinSatTerms(self.field, {})

        terms = {}
        for key, term in self.terms.items():
            terms[key] = LinSatTerm(term.coef * coef, term.var)

        return LinSatTerms(self.field, terms)

    def __rmul__(self, coef: int | FiniteRingElement) -> LinSatTerms:
        return self * coef

    def __neg__(self) -> LinSatTerms:
        return (-1) * self

    def __add__(self, other: LinSatExpr) -> LinSatTerms:
        other = LinSatTerms.from_any(self.field, other)
        return self.add_term_list(other.terms.values())

    def __radd__(self, other: LinSatExpr) -> LinSatTerms:
        return self + other

    def __sub__(self, other: LinSatExpr) -> LinSatTerms:
        return self + (-other)

    def __rsub__(self, other: LinSatExpr) -> LinSatTerms:
        return -self + other

    def __eq__(self, other: LinSatExpr) -> LinSatConstraint:
        if isinstance(other, set):
            constraint = self == 0
            rhs = next(iter(constraint.rhs))
            new_rhs = {el + rhs for el in other}
            return LinSatConstraint(
                constraint.field, constraint.vars, constraint.coefs, new_rhs
            )

        other = LinSatTerms.from_any(self.field, other)
        total = self - other

        rhs = self.field(0)

        variables = []
        coefs = []

        for key, term in total.terms.items():
            if key is None:
                rhs = -term.coef
            else:
                variables.append(term.var)
                coefs.append(term.coef)

        return LinSatConstraint(self.field, variables, coefs, {rhs})

    def __req__(self, other: LinSatExpr) -> LinSatConstraint:
        return self == other

    def __ne__(self, other: LinSatExpr) -> LinSatConstraint:
        return (self == other).invert()

    def __rne__(self, other: LinSatExpr) -> LinSatConstraint:
        return self != other

    def inequality(
        self, other: LinSatExpr, rhs_filter: Callable[[FiniteRingElement], bool]
    ) -> LinSatConstraint:
        if not (isinstance(other, int) or other in self.field):
            raise ValueError(
                "Only constants are supported as the right-hand side of <, >, >= and <="
            )
        other = self.field(other)
        if other not in self.field.base_ring():
            raise ValueError(
                "For extension fields, <, >, >= and <= support only right-hand side elements from the underlying prime field"
            )

        return self == {el for el in self.field.base_ring() if rhs_filter(int(el))}

    def __lt__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el < other)

    def __gt__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el > other)

    def __le__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el <= other)

    def __ge__(self, other: LinSatExpr) -> LinSatConstraint:
        other = int(other)
        return self.inequality(other, lambda el: el >= other)

    def __str__(self) -> str:
        return " + ".join(str(t) for t in self.terms.values())


def get_first_coef(variables: list[LinSatVar], coefs: list[FiniteRingElement]):
    smallest_id = None
    first_coef = None
    for v, c in zip(variables, coefs):
        if smallest_id is None or v.id < smallest_id:
            smallest_id = v.id
            first_coef = first_coef
    return first_coef


class LinSatConstraint:
    def __init__(
        self,
        field: GF,
        vars: list[LinSatVar],
        coefs: list[FiniteRingElement],
        rhs: dict[FiniteRingElement, int],
        n_equations: int = 1,
    ):
        self.field = field

        if isinstance(rhs, set):
            rhs = dict.fromkeys(rhs, 1)

        assert len(vars) == len(coefs)
        assert len(rhs) == 0 or n_equations >= max(rhs.values())
        assert len(rhs) == 0 or min(rhs.values()) >= 0

        self.n_equations = n_equations

        zero_indices = {i for i, c in enumerate(coefs) if c == field(0)}
        self.vars = [v for i, v in enumerate(vars) if i not in zero_indices]
        coefs = [field(c) for c in coefs if field(c) != field(0)]
        rhs = {field(v): weight for v, weight in rhs.items() if weight != 0}

        first_coef = get_first_coef(self.vars, coefs)
        if first_coef is None:
            self.coefs = coefs
            self.rhs = rhs
        else:
            self.coefs = [c / first_coef for c in coefs]
            self.rhs = {c / first_coef: weight for c, weight in rhs.items()}

        self.is_degenerate = False

        if len(self.rhs) == 0:
            self.is_degenerate = True
        if len(self.rhs) == len(field) and len(set(self.rhs.values())) == 1:
            self.is_degenerate = True
        if len(vars) == 0:
            self.is_degenerate = True

    def show_degeneracy_warnings(self):
        if len(self.rhs) == 0:
            warnings.warn(f"Constraint is always false: {self}")
        if len(self.rhs) == len(self.field) and len(set(self.rhs.values())) == 1:
            warnings.warn(f"Constraint is always true: {self}")

    def get_id(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        return tuple(v.id for v in self.vars), tuple(str(c) for c in self.coefs)

    def merge(self, other: LinSatConstraint) -> LinSatConstraint:
        assert self.get_id() == other.get_id()
        new_rhs = self.rhs.copy()
        for el, weight in other.rhs.items():
            if el not in new_rhs and weight > 0:
                new_rhs[el] = 0
            new_rhs[el] += weight
        return LinSatConstraint(
            self.field,
            list(self.vars),
            list(self.coefs),
            new_rhs,
            self.n_equations + other.n_equations,
        )

    def invert(self) -> LinSatConstraint:
        elements = set(self.field)
        new_rhs = {}
        for el in elements:
            weight = self.rhs[el] if el in self.rhs else 0
            new_weight = self.n_equations - weight
            if new_weight > 0:
                new_rhs[el] = new_weight

        return LinSatConstraint(
            self.field, list(self.vars), list(self.coefs), new_rhs, self.n_equations
        )

    def scale_weight(self, scalar: int) -> LinSatConstraint:
        new_rhs = {el: weight * int(scalar) for el, weight in self.rhs.items()}
        return LinSatConstraint(
            self.field, self.vars, self.coefs, new_rhs, self.n_equations * scalar
        )

    def __str__(self) -> str:
        output = []
        weight_to_rhs = {}
        for el, weight in self.rhs.items():
            if weight not in weight_to_rhs:
                weight_to_rhs[weight] = set()
            weight_to_rhs[weight].add(el)

        for weight, rhs in weight_to_rhs.items():
            line = []
            for c, v in zip(self.coefs, self.vars):
                line.append(str(c))
                line.append(str(v))
                line.append("+")

            if len(line) == 0:
                line = ["0"]
            else:
                line.pop()

            if len(rhs) == 1:
                line.append("=")
                line.append(str(next(iter(rhs))))
            else:
                line.append("∈")
                line.append(str(rhs))
            if weight != 1:
                line.append(f"(x{weight})")
            output.append(" ".join(line))

        return "\n".join(output)
