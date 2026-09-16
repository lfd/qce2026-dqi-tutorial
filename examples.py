from __future__ import annotations

import random

import networkx as nx
from sage.all import GF

from classical_solvers import SimAnnealSolver
from dqi import Dqi
from max_constraint_sat import MaxConstraintSat
from max_lin_sat import MaxLinSat


# Knapsack
class Item:
    def __init__(self, weight: int, value: int):
        self.weight = weight
        self.value = value


rng = random.Random(0)

items = [Item(weight=rng.randint(0, 2), value=rng.randint(1, 3)) for _ in range(3)]
W = sum(item.weight for item in items) // 2

knapsack = MaxConstraintSat()
x = [knapsack.new_binary_var(f"x_{i}") for i in range(len(items))]
knapsack.add_constraint(
    sum(x_i * i.weight for x_i, i in zip(x, items)) <= W,
    weight=2 * sum(i.value for i in items),
)
knapsack.add_objective(sum(x_i * i.value for x_i, i in zip(x, items)))


# Minimum vertex cover
n = 10
G = nx.cycle_graph(n)
vertex_cover = MaxConstraintSat()
x = [vertex_cover.new_binary_var(f"x_{i}") for i in range(n)]

vertex_cover.add_objective(
    sum(x_i for x_i in x),
    minimize=True,
)
for u, v in G.edges:
    vertex_cover.add_boolean_constraint(x[u] | x[v], weight=2)


# Maximum 3-colorable subgraph
n = 10
G = nx.erdos_renyi_graph(n, 0.4, seed=0)
colorable_subgraph = MaxLinSat(GF(3))
x = [colorable_subgraph.new_var(f"x_{i}") for i in range(n)]
for u, v in G.edges:
    colorable_subgraph.add_constraint(x[u] != x[v])

for name, prob in [
    ("Knapsack", knapsack),
    ("Minimum Vertex Cover", vertex_cover),
    ("Maximum 3-colorable subgraph", colorable_subgraph),
]:
    print(name)

    sim_anneal = SimAnnealSolver(prob)
    print("simulated annealing:", sim_anneal.get_solution_quality())

    dqi = Dqi(prob)
    print("DQI:", dqi.estimate_solution_quality(1))
