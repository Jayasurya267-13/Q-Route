# QRouteX — Core Prototype Code

## Python backend (offline benchmarking + CLI demo)
- `vrp_core.py` — instance generator, traffic-aware weighted graph, decode/repair, fitness
- `algorithms.py` — QPSO, PSO, GA (order-crossover), nearest-neighbor baseline
- `run_benchmark.py` — runs the full fair-benchmarking protocol (12 runs x 3 algos x 2 sizes), writes results.json
- `demo_cli.py` — single-run CLI: `python3 demo_cli.py --nodes 20 --algorithm all --runs 5 --hybrid`
- `results.json` — the real measured results already generated

## Browser dashboard
- `../qroutex_dashboard.html` — standalone interactive prototype (open directly in any browser, no server needed).
  Runs the same QPSO/PSO/GA logic live in JavaScript, lets you generate instances,
  tune population/iterations, toggle 2-opt hybridization, and see routes on a map +
  convergence chart + results table update in real time.

## Requirements
- Python backend: Python 3 + NumPy only
- Dashboard: any modern browser, internet connection for the Chart.js CDN script tag
  (swap in a local copy of Chart.js if presenting somewhere offline)

## To extend
- Increase INSTANCE_SIZES / N_RUNS in run_benchmark.py for a deeper study
- Swap generate_instance() for a real OSMnx-derived road network + SUMO traffic feed
- Add OR-Tools as a fourth baseline once environment allows installing it
