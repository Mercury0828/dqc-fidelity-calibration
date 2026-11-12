"""Verify, cost, and simulate the calibration plan.

  python run.py plan       verify every circuit, compile to native gates, write results/native_counts.csv
  python run.py submit     submit the representative circuits to IonQ's ideal and forte-1 noisy
                           simulators (native gateset, 1,000 shots); job IDs -> results/simulator_jobs.json
  python run.py collect    fetch the simulator jobs and score them -> results/simulator_results.csv

`submit` and `collect` need the environment variable IONQ_API_KEY. Simulator jobs are free.

Cost formula: IonQ Research Credits resource estimator for Forte
(https://www.ionq.com/programs/research-credits/resource-estimator):
    job size = (N1q / 6.8148 + N2q) * shots, at least 23,000 with debiasing off (150,000 with it on)
    price    = job size * $0.0011213
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import circuits as C  # noqa: E402
import native as N  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
JOBS = RESULTS / "simulator_jobs.json"
SHOTS = 1000
SEED = 20261109


def price(n1, n2, shots=SHOTS, debias=False):
    size = max((n1 / 6.8148 + n2) * shots, 150000 if debias else 23000)
    return size * 0.0011213


# ----------------------------------------------------------------------------- plan
def cmd_plan():
    C.verify_all()
    N.self_test()
    rows = C.plan()
    out = []
    for r in rows:
        nat = N.to_native(r["qc"])
        N.check(r["qc"], nat, [q for q, _ in r["readout"]])
        n1, n2 = N.gate_counts(nat)
        out.append(dict(experiment=r["exp"], circuit=r["name"], qubits=r["qc"].num_qubits, native_1q=n1,
                        native_2q=n2, shots=SHOTS, runs=r["repeats"], price_per_run=round(price(n1, n2), 2),
                        price_total=round(price(n1, n2) * r["repeats"], 2)))
    with open(RESULTS / "native_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    tot = {e: round(sum(o["price_total"] for o in out if o["experiment"] == e), 2) for e in ("E1", "E2")}
    print(f"{len(out)} circuits, {sum(o['runs'] for o in out)} jobs, {sum(o['runs'] for o in out) * SHOTS} shots")
    print(f"Experiment 1 ${tot['E1']:,.2f}  Experiment 2 ${tot['E2']:,.2f}  total ${tot['E1'] + tot['E2']:,.2f}")


# ----------------------------------------------------------------------------- simulators
def backend():
    from qiskit_ionq import IonQProvider
    key = os.environ.get("IONQ_API_KEY")
    if not key:
        sys.exit("Set IONQ_API_KEY (cloud.ionq.com -> API Keys).")
    return IonQProvider(token=key).get_backend("simulator", gateset="native")


def cmd_submit():
    b = backend()
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else {}
    for r in C.simulator_set():
        nat = N.to_native(r["qc"])
        for model in ["ideal", "forte-1"]:
            key = f"{r['name']}|{model}"
            if key in jobs:
                continue
            kw = dict(shots=SHOTS, noise_model=model)
            if model != "ideal":
                kw["sampler_seed"] = SEED
            jobs[key] = b.run(nat, name=f"{r['name']}-{model}", **kw).job_id()
            print(key, jobs[key])
            JOBS.write_text(json.dumps(jobs, indent=1))


def _wait(client, job_id, timeout=3600):
    t0 = time.time()
    while True:
        raw = client.retrieve_job(job_id)
        if raw.get("status") in ("completed", "failed", "canceled") or time.time() - t0 > timeout:
            return raw
        time.sleep(10)


def hellinger(p, q):
    return sum(np.sqrt(p.get(k, 0) * q.get(k, 0)) for k in set(p) | set(q)) ** 2


def _resample(counts, rng):
    keys = list(counts)
    p = np.array([counts[k] for k in keys], float)
    draw = rng.multinomial(int(p.sum()), p / p.sum())
    return {k: int(x) for k, x in zip(keys, draw) if x}


def _probs(counts):
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()}


def score_pack(counts, readout, ideals):
    """Probability of the correct outcome, averaged over copies whose ideal outcome is deterministic."""
    pr = _probs(counts)
    vals = []
    for (q, _), im in zip(readout, ideals):
        top = max(im, key=im.get)
        if im[top] > 0.99:
            vals.append(C.marginal(pr, q).get(top, 0.0))
    p = float(np.mean(vals))
    n = SHOTS * len(vals)
    return dict(metric="success probability", value=round(p, 4), stderr=round(float(np.sqrt(max(p * (1 - p), 0.0) / n)), 4))


def score_circuit(counts, q, ideal, rng, n_boot=2000):
    """Hellinger fidelity to the ideal distribution, with a bootstrap bias correction (the plug-in
    estimate is biased low when the ideal output is spread over many bitstrings)."""
    f = lambda c: hellinger(C.marginal(_probs(c), q), ideal)  # noqa: E731
    point = f(counts)
    boots = np.array([f(_resample(counts, rng)) for _ in range(n_boot)])
    return dict(metric="Hellinger fidelity (bias-corrected)", value=round(2 * point - boots.mean(), 4),
                stderr=round(float(boots.std()), 4), plug_in=round(point, 4))


def cmd_collect():
    b = backend()
    jobs = json.loads(JOBS.read_text())
    rng = np.random.default_rng(SEED)
    rows = []
    for r in C.simulator_set():
        nat = N.to_native(r["qc"])
        n1, n2 = N.gate_counts(nat)
        ideals = [C.ideal_marginal(r["qc"], q) for q, _ in r["readout"]]
        for model in ["ideal", "forte-1"]:
            jid = jobs[f"{r['name']}|{model}"]
            raw = _wait(b.client, jid)
            counts = b.retrieve_job(jid).result().get_counts()
            if r["name"].startswith("sim-"):
                s = score_pack(counts, r["readout"], ideals)
            else:
                s = score_circuit(counts, r["readout"][0][0], ideals[0], rng)
            rows.append(dict(circuit=r["name"], simulator=model, job_id=jid, status=raw.get("status"),
                             qubits=r["qc"].num_qubits, native_1q=n1, native_2q=n2, **s))
    keys = ["circuit", "simulator", "job_id", "status", "qubits", "native_1q", "native_2q", "metric", "value",
            "stderr", "plug_in"]
    with open(RESULTS / "simulator_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    for r in rows:
        print(f"{r['circuit']:24s} {r['simulator']:8s} {r['value']:.4f} +/- {r['stderr']:.4f}  {r['job_id']}")


if __name__ == "__main__":
    {"plan": cmd_plan, "submit": cmd_submit, "collect": cmd_collect}[sys.argv[1]]()
