# Certified attack-cost routing in LEO satellite networks

Research software for computing routing splits and evaluating capacity certificates
under a fixed topology and fluid demand model. Throughputs are in Mbps unless noted.

## Reproduce the analysis

Use Python 3.12 and run these commands from the repository root.

~~~sh
python -m pip install -r requirements.txt
python paper/scripts/make_tables.py results/step5 tables
python paper/scripts/make_figures.py results/step5 figures
python paper/scripts/paper_numbers.py results/step5 summary.json
~~~

The archive contains network-level measurements for 361 topology snapshots and
loss summaries for five snapshots. JSON null in a successful routing's certificate
field represents an infinite model certificate. Check the routing status before
interpreting a missing certificate. Loss inputs are recorded outcomes.

## Evaluate routing

The solver in probe/cac_core.py supports minimum-utilization routing, certified
routing and lexicographic refinement. To run the solver on the synthetic example:

~~~sh
python probe/ts_eval.py examples/snap_0.npz example-results.json 0.001 20 1 1
~~~

The example has three orbital planes with four nodes each. It exercises the input
schema and solver using artificial demands. The paper figures use the measured
summaries in results/. The original topology files are external inputs.

The snapshot adapter in probe/skyfall_snapshot.py reads the SKYFALL configuration,
orbital positions, gateway metadata and cell weights. The adapter is derived from
SpaceNetLab's SKYFALL implementation (commit 17c0888). Its MIT license and
copyright notice are in notices/SKYFALL-LICENSE.txt.

The delay and tolerance diagnostics are in probe/reorder_spread.py and
probe/inf_margin_check.py. To count model dimensions, run
python probe/lp_size.py RESULTS_DIRECTORY with snapshot files under its ts/ directory.
