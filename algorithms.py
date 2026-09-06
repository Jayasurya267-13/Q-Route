"""
QPSO, PSO, and GA for the VRP -- all run under the identical fair-benchmarking
protocol: same population size, same iteration budget, same instance, same
seed schedule. QPSO and PSO work in continuous random-key space and decode
each candidate; GA works natively on permutations (order crossover), matching
the architecture described in the discrete-encoding diagram.
"""
import numpy as np
from vrp_core import fitness_from_keys, fitness_from_permutation


def run_pso(inst, pop_size, iters, seed):
    rng = np.random.default_rng(seed)
    n = inst["n"]
    x = rng.uniform(0, 1, size=(pop_size, n))
    v = rng.uniform(-0.1, 0.1, size=(pop_size, n))

    pbest = x.copy()
    pbest_fit = np.array([fitness_from_keys(x[i], inst)[0] for i in range(pop_size)])
    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    w, c1, c2 = 0.7, 1.5, 1.5
    curve = [gbest_fit]
    for _ in range(iters):
        r1 = rng.uniform(0, 1, size=(pop_size, n))
        r2 = rng.uniform(0, 1, size=(pop_size, n))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, 0, 1)
        fit = np.array([fitness_from_keys(x[i], inst)[0] for i in range(pop_size)])
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        best_idx = np.argmin(pbest_fit)
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit = pbest_fit[best_idx]
            gbest = pbest[best_idx].copy()
        curve.append(gbest_fit)

    _, routes = fitness_from_keys(gbest, inst)
    return gbest_fit, routes, curve


def run_qpso(inst, pop_size, iters, seed):
    """Standard QPSO (Sun et al.) -- delta-potential-well update, no velocity."""
    rng = np.random.default_rng(seed)
    n = inst["n"]
    x = rng.uniform(0, 1, size=(pop_size, n))

    pbest = x.copy()
    pbest_fit = np.array([fitness_from_keys(x[i], inst)[0] for i in range(pop_size)])
    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    curve = [gbest_fit]
    beta_start, beta_end = 1.0, 0.4  # contraction-expansion coefficient, anneals over run
    for it in range(iters):
        beta = beta_start - (beta_start - beta_end) * (it / max(1, iters - 1))
        mbest = pbest.mean(axis=0)
        phi = rng.uniform(0, 1, size=(pop_size, n))
        p = phi * pbest + (1 - phi) * gbest
        u = rng.uniform(1e-6, 1 - 1e-6, size=(pop_size, n))
        sign = rng.choice([-1.0, 1.0], size=(pop_size, n))
        x = p + sign * beta * np.abs(mbest - x) * np.log(1.0 / u)
        x = np.clip(x, 0, 1)

        fit = np.array([fitness_from_keys(x[i], inst)[0] for i in range(pop_size)])
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        best_idx = np.argmin(pbest_fit)
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit = pbest_fit[best_idx]
            gbest = pbest[best_idx].copy()
        curve.append(gbest_fit)

    _, routes = fitness_from_keys(gbest, inst)
    return gbest_fit, routes, curve


def _order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = [None] * n
    child[a:b] = p1[a:b]
    fill = [c for c in p2 if c not in child[a:b]]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def run_ga(inst, pop_size, iters, seed):
    rng = np.random.default_rng(seed)
    n = inst["n"]
    base = np.arange(1, n + 1)
    pop = [rng.permutation(base).tolist() for _ in range(pop_size)]
    fit = np.array([fitness_from_permutation(ind, inst)[0] for ind in pop])

    best_idx = np.argmin(fit)
    gbest, gbest_fit = pop[best_idx][:], fit[best_idx]
    curve = [gbest_fit]

    mutation_rate = 0.15
    tournament_k = 3
    for _ in range(iters):
        new_pop = [gbest[:]]  # elitism: carry the best forward unchanged
        while len(new_pop) < pop_size:
            cand_idx = rng.choice(pop_size, size=tournament_k, replace=False)
            p1 = pop[cand_idx[np.argmin(fit[cand_idx])]]
            cand_idx = rng.choice(pop_size, size=tournament_k, replace=False)
            p2 = pop[cand_idx[np.argmin(fit[cand_idx])]]
            child = _order_crossover(p1, p2, rng)
            if rng.uniform() < mutation_rate:
                i, j = rng.choice(n, size=2, replace=False)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop = new_pop
        fit = np.array([fitness_from_permutation(ind, inst)[0] for ind in pop])
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = pop[best_idx][:]
        curve.append(gbest_fit)

    _, routes = fitness_from_permutation(gbest, inst)
    return gbest_fit, routes, curve


def nearest_neighbor_baseline(inst):
    """Deterministic classical construction heuristic (no metaheuristic) used
    as a simple sanity-check baseline, not as a 'weak strawman' comparison."""
    n = inst["n"]
    unvisited = set(range(1, n + 1))
    order = []
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: inst["W"][current, c])
        order.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    fit, routes = fitness_from_permutation(order, inst)
    return fit, routes
