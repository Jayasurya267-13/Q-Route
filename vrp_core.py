"""
QRouteX benchmarking core.
Defines the VRP instance, the traffic-aware weighted graph, the decode
(continuous key -> discrete route) mechanism, and the shared fitness function
used identically by every algorithm being compared.
"""
import numpy as np

# ---- Objective weights (must match the mathematical formulation slide) ----
ALPHA, BETA, GAMMA = 0.35, 0.45, 0.20      # weights inside W_ij(t) = aD + bT + gC
W1, W2, W3 = 0.4, 0.35, 0.25               # weights inside F = w1*T + w2*D + w3*C
SPEED = 40.0                               # nominal speed used to convert distance -> time


def generate_instance(seed, n_customers, n_vehicles, capacity, coord_range=100.0,
                       demand_low=3, demand_high=10):
    """Synthetic VRP instance: depot (index 0) + n_customers, with a
    static-but-nontrivial congestion field over every edge (models 'current'
    traffic conditions the way SUMO/synthetic traffic would)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, coord_range, size=(n_customers + 1, 2))
    demands = rng.integers(demand_low, demand_high + 1, size=n_customers + 1).astype(float)
    demands[0] = 0.0

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # D_ij

    congestion = rng.uniform(0.8, 1.6, size=dist.shape)
    congestion = (congestion + congestion.T) / 2.0
    np.fill_diagonal(congestion, 0.0)  # C_ij

    time_mat = dist * congestion / SPEED  # T_ij: congestion inflates travel time

    W = ALPHA * dist + BETA * time_mat + GAMMA * congestion  # W_ij(t), the routing cost

    return dict(
        n=n_customers, n_vehicles=n_vehicles, capacity=capacity,
        coords=coords, demands=demands,
        dist=dist, time=time_mat, congestion=congestion, W=W,
    )


def decode(keys, inst):
    """Continuous random-key vector (length n_customers, values in [0,1])
    -> discrete route(s) respecting vehicle capacity.
    Step 1: argsort keys -> giant tour (permutation of customers).
    Step 2: greedy split into vehicle routes by capacity.
    Step 3: repair -- if more routes are needed than vehicles available,
    the overflow is merged into the last vehicle and a capacity-violation
    penalty is charged (soft constraint), matching the repair mechanism
    described in the discrete-encoding diagram."""
    order = np.argsort(keys) + 1  # customer ids, 1..n
    return split_by_capacity(order, inst)


def split_by_capacity(order, inst):
    demands = inst["demands"]
    cap = inst["capacity"]
    n_vehicles = inst["n_vehicles"]

    routes, current, load = [], [], 0.0
    for c in order:
        d = demands[int(c)]
        if load + d > cap and current:
            routes.append(current)
            current, load = [], 0.0
        current.append(int(c))
        load += d
    if current:
        routes.append(current)

    penalty = 0.0
    if len(routes) > n_vehicles:
        # repair: merge overflow routes into the last permitted vehicle
        keep, overflow = routes[:n_vehicles - 1], routes[n_vehicles - 1:]
        merged = [c for r in overflow for c in r]
        keep.append(merged)
        routes = keep
        # soft penalty proportional to how far the merged route overshoots capacity
        overshoot = max(0.0, sum(demands[c] for c in merged) - cap)
        penalty = 50.0 * overshoot
    return routes, penalty


def route_cost(route, inst, matrix_key="W"):
    """Cost of a single depot->...->depot route under a given cost matrix."""
    M = inst[matrix_key]
    if not route:
        return 0.0
    total = M[0, route[0]]
    for a, b in zip(route[:-1], route[1:]):
        total += M[a, b]
    total += M[route[-1], 0]
    return total


def solution_metrics(routes, inst):
    """Real distance / time / congestion totals for reporting (not just the
    scalarized fitness) -- used for the benchmark metrics table."""
    dist = sum(route_cost(r, inst, "dist") for r in routes)
    time_ = sum(route_cost(r, inst, "time") for r in routes)
    cong = sum(route_cost(r, inst, "congestion") for r in routes)
    return dist, time_, cong


def fitness_from_keys(keys, inst):
    routes, penalty = decode(keys, inst)
    dist, time_, cong = solution_metrics(routes, inst)
    F = W1 * time_ + W2 * dist + W3 * cong + penalty
    return F, routes


def fitness_from_permutation(perm, inst):
    routes, penalty = split_by_capacity(perm, inst)
    dist, time_, cong = solution_metrics(routes, inst)
    F = W1 * time_ + W2 * dist + W3 * cong + penalty
    return F, routes


def two_opt(route, inst, matrix_key="W", max_passes=1):
    """One (or a few) full 2-opt improvement passes on a single route,
    used only in the separate hybridization experiment -- kept OUT of the
    main QPSO/PSO/GA comparison so that comparison isolates the global
    search algorithm itself (fair-benchmarking protocol)."""
    if len(route) < 3:
        return route
    best = route[:]
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_cost(new_route, inst, matrix_key) < route_cost(best, inst, matrix_key) - 1e-9:
                    best = new_route
                    improved = True
    return best
