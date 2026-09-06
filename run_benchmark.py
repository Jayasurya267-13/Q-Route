import json
import time
import numpy as np

from vrp_core import generate_instance, solution_metrics, two_opt
from algorithms import run_pso, run_qpso, run_ga, nearest_neighbor_baseline

POP_SIZE = 30
ITERS = 150
N_RUNS = 12          # independent runs per algorithm per instance (fair-benchmarking protocol)
INSTANCE_SIZES = [20, 40]   # nodes, for the scalability trend
CAPACITY = 45
CAPACITY_SAFETY_MARGIN = 1.2   # total fleet capacity = margin x total demand, so the instance stays feasible
INSTANCE_SEED = 7    # same instance (same nodes/demands/congestion) for every algorithm

ALGOS = {"QPSO": run_qpso, "PSO": run_pso, "GA": run_ga}


def run_all():
    results = {}
    for n_customers in INSTANCE_SIZES:
        inst = generate_instance(INSTANCE_SEED, n_customers, n_vehicles=1, capacity=CAPACITY)
        total_demand = inst["demands"].sum()
        n_vehicles = max(2, int(np.ceil(total_demand * CAPACITY_SAFETY_MARGIN / CAPACITY)))
        inst["n_vehicles"] = n_vehicles

        size_key = f"{n_customers}_nodes"
        results[size_key] = {"instance": {
            "n_customers": n_customers, "n_vehicles": n_vehicles, "capacity": CAPACITY,
            "total_demand": float(total_demand),
        }}

        # classical deterministic baseline (nearest-neighbor construction)
        nn_fit, nn_routes = nearest_neighbor_baseline(inst)
        nn_dist, nn_time, nn_cong = solution_metrics(nn_routes, inst)
        results[size_key]["NearestNeighbor"] = {
            "fitness": nn_fit, "distance": nn_dist, "time": nn_time, "congestion": nn_cong,
        }

        for name, fn in ALGOS.items():
            run_fits, run_dists, run_times, run_congs, run_runtimes = [], [], [], [], []
            curves = []
            for run_idx in range(N_RUNS):
                seed = 1000 * (list(ALGOS).index(name) + 1) + run_idx  # distinct but reproducible per algo/run
                t0 = time.perf_counter()
                fit, routes, curve = fn(inst, POP_SIZE, ITERS, seed)
                runtime = time.perf_counter() - t0

                dist, time_, cong = solution_metrics(routes, inst)
                run_fits.append(fit)
                run_dists.append(dist)
                run_times.append(time_)
                run_congs.append(cong)
                run_runtimes.append(runtime)
                curves.append(curve)

            curves = np.array(curves)  # (N_RUNS, ITERS+1)
            mean_curve = curves.mean(axis=0).tolist()

            # iteration at which the mean curve first reaches within 1% of its own final value
            final_val = mean_curve[-1]
            threshold = final_val * 1.01 if final_val >= 0 else final_val * 0.99
            conv_iter = next((i for i, v in enumerate(mean_curve) if v <= threshold), len(mean_curve) - 1)

            results[size_key][name] = {
                "fitness_mean": float(np.mean(run_fits)), "fitness_std": float(np.std(run_fits)),
                "distance_mean": float(np.mean(run_dists)), "distance_std": float(np.std(run_dists)),
                "time_mean": float(np.mean(run_times)), "time_std": float(np.std(run_times)),
                "congestion_mean": float(np.mean(run_congs)), "congestion_std": float(np.std(run_congs)),
                "runtime_sec_mean": float(np.mean(run_runtimes)), "runtime_sec_std": float(np.std(run_runtimes)),
                "convergence_iteration": int(conv_iter),
                "mean_convergence_curve": mean_curve,
                "n_runs": N_RUNS, "pop_size": POP_SIZE, "iterations": ITERS,
            }

        # Hybridization experiment: apply the SAME 2-opt post-processing to
        # every algorithm's best solution (not just QPSO) -- otherwise the
        # hybrid comparison would be cherry-picked in QPSO's favor.
        from vrp_core import route_cost
        hybrid_seed = 1000
        results[size_key]["hybrid_2opt"] = {}
        for name, fn in ALGOS.items():
            algo_offset = 1000 * (list(ALGOS).index(name) + 1)
            fit, routes, _ = fn(inst, POP_SIZE, ITERS, seed=algo_offset)
            improved_routes = [two_opt(r, inst, "W", max_passes=3) for r in routes]
            before = sum(route_cost(r, inst, "W") for r in routes)
            after = sum(route_cost(r, inst, "W") for r in improved_routes)
            results[size_key]["hybrid_2opt"][name] = {
                "fitness_before": before, "fitness_after": after,
                "improvement_pct": 100.0 * (before - after) / before if before else 0.0,
            }

    return results


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = run_all()
    total_time = time.perf_counter() - t0
    results["_meta"] = {
        "total_wall_time_sec": total_time,
        "protocol": {
            "population_size": POP_SIZE, "iterations": ITERS, "independent_runs": N_RUNS,
            "instance_sizes": INSTANCE_SIZES, "same_instance_across_algorithms": True,
            "same_seed_schedule_across_algorithms": "per-run seeds fixed per algorithm index",
        },
    }
    with open("/home/claude/qroutex_bench/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done in {total_time:.1f}s")
    for size_key in [k for k in results if k != "_meta"]:
        print(f"\n== {size_key} ==")
        for algo in ["NearestNeighbor", "QPSO", "PSO", "GA"]:
            r = results[size_key][algo]
            if "fitness_mean" in r:
                print(f"{algo:16s} F={r['fitness_mean']:.2f}±{r['fitness_std']:.2f}  "
                      f"dist={r['distance_mean']:.1f}  time={r['time_mean']:.2f}  "
                      f"cong={r['congestion_mean']:.1f}  runtime={r['runtime_sec_mean']:.3f}s  "
                      f"conv_iter={r['convergence_iteration']}")
            else:
                print(f"{algo:16s} F={r['fitness']:.2f}  dist={r['distance']:.1f}  "
                      f"time={r['time']:.2f}  cong={r['congestion']:.1f}")
