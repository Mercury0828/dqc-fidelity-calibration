# Calibrating the Fidelity Cost of Qubit Migration and Remote Gates

Code for the IonQ Research Credits project by Hui Zhong (Miami University) and Jiachen Shen (University of Houston).

The project treats one 36-qubit IonQ Forte chip as two or three virtual QPUs. It measures the fidelity cost of two distributed-computing primitives:

- **Migration.** A data qubit is teleported to a fresh qubit on another virtual QPU.
- **Remote gate.** A two-qubit gate between virtual QPUs uses the cat-entangler / cat-disentangler protocol (Eisert et al., PRA 62, 052317, 2000).

IonQ targets do not branch on mid-circuit measurement results. Every measurement-conditioned correction is therefore written as a controlled gate (deferred measurement).

## Files

| File | Contents |
|---|---|
| `circuits.py` | Builds every circuit and verifies it with an ideal simulator |
| `native.py` | Compiles circuits to IonQ native gates (GPi, GPi2, ZZ) and checks the result |
| `run.py` | `plan`, `submit`, and `collect` commands |
| `results/native_counts.csv` | Native gate counts and estimated price for all 52 planned circuits |
| `results/simulator_jobs.json` | IonQ simulator job IDs |
| `results/simulator_results.csv` | Scores of the simulator jobs |

## Experiments

**Experiment 1: primitive calibration.**

- Teleportation copies (3 qubits) cover 6 input states and 3 measurement bases.
- Remote-CNOT copies (4 qubits) cover 4 truth-table inputs, plus 2 Bell-state inputs in 3 parity bases.
- Each setting also runs with X, Y, or Z applied to one half of the EPR pair. These Paulis map |Φ+⟩ to the other three Bell states, so a weighted mix of the four results gives the outcome for any Bell-diagonal pair.
- Each 36-qubit circuit holds 12 teleportations or 9 remote CNOTs.

**Experiment 2: predictive validation.**

- A 6-qubit QFT, whose input is prepared so that the ideal output is a single bitstring.
- A 6-qubit 1D transverse-field Ising circuit with 4 Trotter steps.
- Each circuit runs monolithically, and split over 2 or 3 virtual QPUs. A split circuit uses either a static partition (one fresh EPR pair per cross-partition gate) or one mid-run migration that minimises the remaining remote gates.

**Plan.** The plan has 52 circuits. Experiment 1 runs three times to measure drift, and Experiment 2 runs twice. That makes 116 jobs at 1,000 shots each, with debiasing off.

## Native compilation

`native.py` transpiles to {rz, sx, x, rzz}. It then maps the gates as follows:

- sx → GPi2
- x → GPi
- rzz → ZZ
- rz is tracked as a phase frame and folded into the next GPi/GPi2 phase.

Each controlled gate therefore costs one ZZ. Two checks confirm the compiled circuits:

- Every compiled circuit reproduces the ideal output distribution of the original circuit (`run.py plan`).
- On IonQ's ideal simulator, every submitted circuit returns its intended output.

## Cost

Prices use the formula of IonQ's Research Credits resource estimator for Forte:

```
job size = max((N1q / 6.8148 + N2q) × shots, 23,000)   # debiasing off
price    = job size × $0.0011213
```

With the native counts in `results/native_counts.csv`, the plan costs $7,313.58. Of that, Experiment 1 is $2,277.72 and Experiment 2 is $5,035.86.

## Usage

```
pip install -r requirements.txt
python run.py plan                       # no account needed
export IONQ_API_KEY=...                  # PowerShell: $env:IONQ_API_KEY = "..."
python run.py submit                     # free simulator jobs (skips jobs already in simulator_jobs.json)
python run.py collect
```

The IonQ simulators accept at most 29 qubits. The simulator versions of the Experiment 1 packs therefore hold 9 teleportations (27 qubits) and 7 remote CNOTs (28 qubits).

## Simulator results (forte-1 noise model, 1,000 shots)

| Circuit | Qubits | Native 1q / 2q | Result | Noisy-sim job ID |
|---|---|---|---|---|
| Teleportation pack | 27 | 72 / 36 | 0.964 ± 0.003 success | 01a1c1b7-ac90-7606-85ce-31412a2bcb46 |
| Remote-CNOT pack | 28 | 73 / 35 | 0.952 ± 0.003 success | 01a1c1b7-b1cb-7629-a90d-9c3f05cbe50c |
| QFT-6, monolithic | 6 | 11 / 15 | 0.809 ± 0.012 | 01a1c1b7-b796-77ca-95f2-ab62f1ed123b |
| QFT-6, split, static | 24 | 77 / 51 | 0.574 ± 0.016 | 01a1c1b7-bcfd-75e8-89e4-3305b505617b |
| QFT-6, split, one migration | 20 | 63 / 43 | 0.630 ± 0.015 | 01a1c1b7-c278-73e4-994c-8330cc6cb0b6 |
| TFIM-6, monolithic | 6 | 48 / 20 | 0.968 ± 0.007 | 01a1c1b7-c82d-72be-845e-630737b536fe |
| TFIM-6, split, static | 14 | 83 / 36 | 0.940 ± 0.009 | 01a1c1b7-cdaa-727f-99ae-f92a3a2cdd1d |
| TFIM-6, split, one migration | 16 | 89 / 40 | 0.942 ± 0.008 | 01a1c1b7-d33e-7135-8939-3635854abb5c |

**How to read the results:**

- For the two packs, "success" is the probability of the correct outcome. It is averaged over the copies whose ideal outcome is deterministic.
- The other rows give Hellinger fidelity to the exact ideal distribution, with a bootstrap bias correction.
- The ± values are standard errors from 1,000 shots.
- The ideal-simulator job IDs are listed in `results/simulator_jobs.json` and `results/simulator_results.csv`.

## License

MIT. If you use this code, please acknowledge: "This material is supported via compute credits from IonQ."
