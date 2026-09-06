#!/usr/bin/env python3
"""
QRouteX core prototype -- CLI demo.

Takes instance parameters, runs the chosen algorithm(s), and prints the
resulting routes + real metrics. Optionally writes a JSON file the HTML
dashboard can be seeded from for a matching side-by-side comparison.

Usage examples:
  python3 demo_cli.py --nodes 20 --algorithm qpso
  python3 demo_cli.py --nodes 30 --algorithm all --runs 5 --hybrid
  python3 demo_cli.py --nodes 20 --algorithm qpso --out result.json
"""
import argparse
import json
import time
import numpy as np

from vrp_core import generate_instance, solution_metrics, two_opt, route_cost
from algorithms import run_pso, run_qpso, run_ga, nearest_neighbor_baseline

ALGOS = {"qpso": run_qpso, "pso": run_pso, "ga": run_ga}


def summarize_instance(inst):
    print(f"Instance: {inst['n']} customers, {inst['n_vehicles']} vehicles, "
          f"capacity {inst['capacity']}, total demand {inst['demands'].sum():.0f}")


def run_one(name, fn, inst, pop, iters, runs, seed_base, hybrid):
    fits, dists, times, congs, runtimes = [], [], [], [], []
    best_routes, best_fit = None, float("inf")
    for r in range(runs):
        t0 = time.perf_counter()
        fit, routes, curve = fn(inst, pop, iters, seed=seed_base + r)
        runtime = time.perf_counter() - t0
        dist, time_, cong = solution_metrics(routes, inst)
        fits.append(fit); dists.append(dist); times.append(time_)
        congs.append(cong); runtimes.append(runtime)
        if fit < best_fit:
            best_fit, best_routes = fit, routes

    hybrid_fit = None
    if hybrid and best_routes is not None:
        improved = [two_opt(r, inst, "W", max_passes=3) for r in best_routes]
        before = sum(route_cost(r, inst, "W") for r in best_routes)
        after = sum(route_cost(r, inst, "W") for r in improved)
        hybrid_fit = after
        best_routes = improved

    print(f"\n[{name.upper()}]  runs={runs}  pop={pop}  iters={iters}")
    print(f"  Fitness F   : {np.mean(fits):.2f} ± {np.std(fits):.2f}")
    print(f"  Distance    : {np.mean(dists):.2f}")
    print(f"  Time        : {np.mean(times):.2f}")
    print(f"  Congestion  : {np.mean(congs):.2f}")
    print(f"  Runtime     : {np.mean(runtimes)*1000:.1f} ms")
    if hybrid_fit is not None:
        print(f"  +2-opt final F: {hybrid_fit:.2f}")
    print("  Best routes :")
    for i, r in enumerate(best_routes, 1):
        print(f"    Vehicle {i}: Depot -> {' -> '.join(map(str, r))} -> Depot")

    return {
        "algorithm": name, "fitness_mean": float(np.mean(fits)), "fitness_std": float(np.std(fits)),
        "distance_mean": float(np.mean(dists)), "time_mean": float(np.mean(times)),
        "congestion_mean": float(np.mean(congs)), "runtime_ms_mean": float(np.mean(runtimes) * 1000),
        "hybrid_final_fitness": hybrid_fit, "best_routes": best_routes,
    }


def main():
    ap = argparse.ArgumentParser(description="QRouteX core prototype CLI demo")
    ap.add_argument("--nodes", type=int, default=20, help="number of customer nodes")
    ap.add_argument("--capacity", type=float, default=45, help="vehicle capacity")
    ap.add_argument("--seed", type=int, default=7, help="instance seed")
    ap.add_argument("--algorithm", choices=["qpso", "pso", "ga", "all"], default="qpso")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--runs", type=int, default=1, help="independent runs to average over")
    ap.add_argument("--hybrid", action="store_true", help="apply 2-opt to the best route found")
    ap.add_argument("--out", type=str, default=None, help="optional path to write JSON summary")
    args = ap.parse_args()

    inst = generate_instance(args.seed, args.nodes, n_vehicles=1, capacity=args.capacity)
    total_demand = inst["demands"].sum()
    inst["n_vehicles"] = max(2, int(np.ceil(total_demand * 1.2 / args.capacity)))
    summarize_instance(inst)

    nn_fit, nn_routes = nearest_neighbor_baseline(inst)
    nn_dist, nn_time, nn_cong = solution_metrics(nn_routes, inst)
    print(f"\n[NEAREST NEIGHBOR baseline]  F={nn_fit:.2f}  dist={nn_dist:.2f}  "
          f"time={nn_time:.2f}  cong={nn_cong:.2f}")

    names = list(ALGOS) if args.algorithm == "all" else [args.algorithm]
    output = {"instance": {"n": inst["n"], "n_vehicles": inst["n_vehicles"], "capacity": inst["capacity"]},
              "baseline_nn": {"fitness": nn_fit, "distance": nn_dist, "time": nn_time, "congestion": nn_cong},
              "results": []}
    for i, name in enumerate(names):
        seed_base = 1000 * (i + 1)
        result = run_one(name, ALGOS[name], inst, args.pop, args.iters, args.runs, seed_base, args.hybrid)
        output["results"].append(result)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
