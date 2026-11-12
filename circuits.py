"""Circuits for calibrating the fidelity cost of qubit migration and remote gates.

One IonQ Forte chip is treated as two or three "virtual QPUs". Qubits on different virtual QPUs
interact only through EPR-assisted protocols:

  * migration   = state teleportation of a data qubit to a fresh qubit on another virtual QPU
  * remote gate = cat-entangler / cat-disentangler (Eisert et al., PRA 62, 052317, 2000)

IonQ targets do not branch on mid-circuit measurement results, so every measurement-conditioned
correction is written in deferred-measurement form (a controlled gate from the would-be-measured
qubit). Every construction is checked against an ideal simulator in `verify_all()`.

Experiment 1 (primitive calibration)
    Teleportation copies (3 qubits) over 6 input states x 3 measurement bases, and remote-CNOT copies
    (4 qubits) over 4 truth-table inputs and 2 Bell-state inputs x 3 parity bases. Each setting also
    runs with X, Y or Z applied to one half of the EPR pair. Because those three Paulis map |Phi+> to
    the other three Bell states, a weighted mix of the four results gives the outcome for any
    Bell-diagonal EPR pair. Copies are packed side by side: 12 teleportations or 9 remote CNOTs per
    36-qubit circuit. One baseline circuit holds local CNOTs and state-preparation-and-measurement.

Experiment 2 (predictive validation)
    A 6-qubit QFT (input prepared so the ideal output is one bitstring) and a 6-qubit 1D
    transverse-field Ising (TFIM) circuit with 4 Trotter steps. Each runs monolithically, and split over
    2 or 3 virtual QPUs either with a static partition (every cross-partition gate is a remote gate on a
    fresh EPR pair) or with one mid-run migration (the single move that minimises the remaining number
    of remote gates). Ancilla qubits are not reused.
"""
from __future__ import annotations

import math
import random

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace

N_QUBITS_QPU = 36       # IonQ Forte
N_LOGICAL = 6           # Experiment 2 circuit size
TROTTER_STEPS = 4
TFIM_FIELDS = [0.5, 1.0, 1.5, 2.0]
LAYOUTS = {2: [0, 0, 0, 1, 1, 1], 3: [0, 0, 1, 1, 2, 2]}   # logical qubit -> virtual QPU
QFT_SEED = 20261106


# ----------------------------------------------------------------------------- small helpers
def prep_1q(qc, q, label):
    """Prepare one of the six Pauli eigenstates."""
    if label in ("1", "-", "-i"):
        qc.x(q)
    if label in ("+", "-", "+i", "-i"):
        qc.h(q)
    if label in ("+i", "-i"):
        qc.s(q)


def rotate_to_basis(qc, q, basis):
    """Rotate so that the final Z measurement measures `basis`."""
    if basis == "X":
        qc.h(q)
    elif basis == "Y":
        qc.sdg(q)
        qc.h(q)


def pauli(qc, q, p):
    {"I": lambda: None, "X": lambda: qc.x(q), "Y": lambda: qc.y(q), "Z": lambda: qc.z(q)}[p]()


def marginal(probs: dict, qubits: list[int]) -> dict:
    """Marginal over `qubits` from a dict keyed by Qiskit bitstrings (qubit 0 rightmost)."""
    out = {}
    for key, p in probs.items():
        bits = key.replace(" ", "")[::-1]
        k = "".join(bits[q] for q in qubits)
        out[k] = out.get(k, 0.0) + p
    return out


def dist_close(a, b, tol=1e-9):
    return all(abs(a.get(k, 0) - b.get(k, 0)) < tol for k in set(a) | set(b))


def ideal_marginal(qc: QuantumCircuit, qubits: list[int]) -> dict:
    """Exact ideal marginal on `qubits` (statevector up to 20 qubits, matrix-product state above)."""
    c = qc.remove_final_measurements(inplace=False)
    if c.num_qubits <= 20:
        return marginal(Statevector.from_instruction(c).probabilities_dict(), qubits)
    from qiskit_aer import AerSimulator
    c.save_probabilities_dict(qubits)
    sim = AerSimulator(method="matrix_product_state")
    data = sim.run(transpile(c, sim, optimization_level=0)).result().data(0)["probabilities"]
    out = {}
    for k, v in data.items():
        k = int(k, 16) if isinstance(k, str) else int(k)
        out["".join(str((k >> j) & 1) for j in range(len(qubits)))] = v
    return out


# ----------------------------------------------------------------------------- Experiment 1
TELEPORT_SETTINGS = [(i, b) for i in ["0", "1", "+", "-", "+i", "-i"] for b in "XYZ"]          # 18
CNOT_SETTINGS = ([((c, t), ("Z", "Z")) for c in "01" for t in "01"]
                 + [((c, "0"), bb) for c in "+-" for bb in [("X", "X"), ("Y", "Y"), ("Z", "Z")]])  # 10
PAULIS = ["I", "X", "Y", "Z"]


def teleport_copy(qc, d, a, b, inp, basis, ins="I"):
    """Move the state on d to b through the EPR pair (a, b)."""
    prep_1q(qc, d, inp)
    qc.h(a); qc.cx(a, b)          # EPR pair standing in for a network-delivered pair
    pauli(qc, b, ins)             # optional Pauli on one half of the pair
    qc.cx(d, a); qc.h(d)          # Bell-basis rotation on the sending side
    qc.cx(a, b); qc.cz(d, b)      # deferred X / Z corrections on the receiving side
    rotate_to_basis(qc, b, basis)


def remote_cnot_copy(qc, c, a, b, t, inp, basis, ins="I"):
    """CNOT(c -> t) with c and t on different virtual QPUs, through the EPR pair (a, b)."""
    prep_1q(qc, c, inp[0]); prep_1q(qc, t, inp[1])
    qc.h(a); qc.cx(a, b)
    pauli(qc, b, ins)
    qc.cx(c, a); qc.cx(a, b)      # cat-entangler (deferred correction)
    qc.cx(b, t)                   # the gate, applied locally on the target's QPU
    qc.h(b); qc.cz(b, c)          # cat-disentangler (deferred correction)
    rotate_to_basis(qc, c, basis[0]); rotate_to_basis(qc, t, basis[1])


def local_cnot_copy(qc, c, t, inp, basis):
    prep_1q(qc, c, inp[0]); prep_1q(qc, t, inp[1])
    qc.cx(c, t)
    rotate_to_basis(qc, c, basis[0]); rotate_to_basis(qc, t, basis[1])


def packed(kind, settings, max_qubits=N_QUBITS_QPU):
    """Pack copies side by side. Returns a list of (circuit, readout) where readout lists
    (measured qubits, setting) for each copy."""
    width = 3 if kind == "teleport" else 4
    per = max_qubits // width
    out = []
    for start in range(0, len(settings), per):
        chunk = settings[start:start + per]
        qc = QuantumCircuit(width * len(chunk))
        readout = []
        for j, (inp, basis, ins) in enumerate(chunk):
            q = list(range(j * width, (j + 1) * width))
            if kind == "teleport":
                teleport_copy(qc, *q, inp, basis, ins)
                readout.append(([q[2]], (inp, basis, ins)))
            else:
                remote_cnot_copy(qc, *q, inp, basis, ins)
                readout.append(([q[0], q[3]], (inp, basis, ins)))
        qc.measure_all()
        out.append((qc, readout))
    return out


def baseline_circuit():
    qc = QuantumCircuit(26)
    readout = []
    for j, (inp, b) in enumerate(CNOT_SETTINGS):
        local_cnot_copy(qc, 2 * j, 2 * j + 1, inp, b)
        readout.append(([2 * j, 2 * j + 1], ("local-cnot", inp, b)))
    for j, (inp, b) in enumerate([("0", "Z"), ("1", "Z"), ("+", "X"), ("-", "X"), ("+i", "Y"), ("-i", "Y")]):
        prep_1q(qc, 20 + j, inp); rotate_to_basis(qc, 20 + j, b)
        readout.append(([20 + j], ("spam", inp, b)))
    qc.measure_all()
    return qc, readout


def experiment1():
    rows = []
    tele = [(i, b, p) for (i, b) in TELEPORT_SETTINGS for p in PAULIS]
    for k, (qc, ro) in enumerate(packed("teleport", tele)):
        rows.append(dict(exp="E1", name=f"teleport-pack{k}", qc=qc, readout=ro, repeats=3))
    rcx = [(i, b, p) for (i, b) in CNOT_SETTINGS for p in PAULIS]
    for k, (qc, ro) in enumerate(packed("rcnot", rcx)):
        rows.append(dict(exp="E1", name=f"remote-cnot-pack{k}", qc=qc, readout=ro, repeats=3))
    qc, ro = baseline_circuit()
    rows.append(dict(exp="E1", name="baseline-pack0", qc=qc, readout=ro, repeats=3))
    return rows


# ----------------------------------------------------------------------------- Experiment 2
def logical_qft(n):
    g = []
    for i in range(n):
        g.append(("h", i))
        for j in range(i + 1, n):
            g.append(("cp", math.pi / 2 ** (j - i), j, i))
    return g


def logical_tfim(n, steps, h, dt=0.25, J=1.0):
    g = []
    for _ in range(steps):
        g += [("rzz", 2 * J * dt, i, i + 1) for i in range(n - 1)]
        g += [("rx", 2 * h * dt, i) for i in range(n)]
    return g


def _apply_local(qc, op, phys):
    kind = op[0]
    if kind == "h":
        qc.h(phys[op[1]])
    elif kind == "rx":
        qc.rx(op[1], phys[op[2]])
    elif kind == "cp":
        qc.cp(op[1], phys[op[2]], phys[op[3]])
    elif kind == "rzz":
        qc.rzz(op[1], phys[op[2]], phys[op[3]])


def _apply_remote(qc, op, phys, a, b):
    kind, theta, c, t = op
    qc.h(a); qc.cx(a, b)                 # EPR pair
    qc.cx(phys[c], a); qc.cx(a, b)       # cat-entangle the control onto b
    if kind == "cp":
        qc.cp(theta, b, phys[t])
    else:                                # rzz = cx . rz . cx with the control copy on b
        qc.cx(b, phys[t]); qc.rz(theta, phys[t]); qc.cx(b, phys[t])
    qc.h(b); qc.cz(b, phys[c])           # cat-disentangle


def _apply_migration(qc, d, a, b):
    """Teleport the qubit on d to b through the EPR pair (a, b)."""
    qc.h(a); qc.cx(a, b)
    qc.cx(d, a); qc.h(d)
    qc.cx(a, b); qc.cz(d, b)


def _remote_count(ops, part):
    return sum(1 for op in ops if op[0] in ("cp", "rzz") and part[op[2]] != part[op[3]])


def _best_migration(ops, part, n_qpu, cap):
    """The single move (qubit, destination) that minimises the remaining remote gates."""
    best = (10 ** 9, None, None)
    for q in range(len(part)):
        for dest in range(n_qpu):
            if dest == part[q] or part.count(dest) >= cap:
                continue
            p2 = list(part); p2[q] = dest
            r = _remote_count(ops, p2)
            if r < best[0]:
                best = (r, q, dest)
    return best


def distributed(ops, part, n_qpu, variant, prep=None):
    """Build the physical circuit. variant: monolithic | static | migrate.
    Returns (circuit, physical homes of the logical qubits at the end)."""
    n = len(part)
    migr = None
    if variant == "migrate":
        half = len(ops) // 2
        _, q, dest = _best_migration(ops[half:], part, n_qpu, cap=math.ceil(n / n_qpu) + 1)
        migr = (half, q, dest)
    part_now, need = list(part), 0
    for idx, op in enumerate(ops):
        if migr and idx == migr[0]:
            need += 2; part_now[migr[1]] = migr[2]
        if variant != "monolithic" and op[0] in ("cp", "rzz") and part_now[op[2]] != part_now[op[3]]:
            need += 2
    assert n + need <= N_QUBITS_QPU
    qc = QuantumCircuit(n + need)
    phys, nxt, part_now = list(range(n)), n, list(part)
    if prep:
        prep(qc, phys)
    for idx, op in enumerate(ops):
        if migr and idx == migr[0]:
            _apply_migration(qc, phys[migr[1]], nxt, nxt + 1)
            phys[migr[1]] = nxt + 1; nxt += 2
            part_now[migr[1]] = migr[2]
        if variant != "monolithic" and op[0] in ("cp", "rzz") and part_now[op[2]] != part_now[op[3]]:
            _apply_remote(qc, op, phys, nxt, nxt + 1); nxt += 2
        else:
            _apply_local(qc, op, phys)
    qc.measure_all()
    return qc, phys


def _qft_input(x, n):
    """Per-qubit (theta, phi) so that ry(theta), rz(phi) prepares QFT^-1|x> (a product state);
    the ideal QFT output is then the single bitstring x."""
    fwd = QuantumCircuit(n)
    for op in logical_qft(n):
        _apply_local(fwd, op, list(range(n)))
    psi = Statevector.from_label(format(x, f"0{n}b")).evolve(fwd.inverse())
    angles = []
    for q in range(n):
        rho = partial_trace(psi, [k for k in range(n) if k != q]).data
        bx, by, bz = 2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real
        angles.append((math.acos(max(-1.0, min(1.0, bz))), math.atan2(by, bx)))
    chk = QuantumCircuit(n)
    for q, (th, ph) in enumerate(angles):
        chk.ry(th, q); chk.rz(ph, q)
    assert abs(np.vdot(Statevector.from_instruction(chk).data, psi.data)) ** 2 > 1 - 1e-9
    return angles


def qft_inputs():
    rng = random.Random(QFT_SEED)
    return [rng.randrange(2 ** N_LOGICAL) for _ in range(4)]


def experiment2():
    rows = []
    for x in qft_inputs():
        angles = _qft_input(x, N_LOGICAL)

        def prep(qc, phys, angles=angles):
            for q, (th, ph) in enumerate(angles):
                qc.ry(th, phys[q]); qc.rz(ph, phys[q])
        variants = [("monolithic", 2, "monolithic")] + [(f"{nq}qpu-{v}", nq, v) for nq in LAYOUTS
                                                      for v in ("static", "migrate")]
        for label, nq, v in variants:
            qc, homes = distributed(logical_qft(N_LOGICAL), LAYOUTS[nq], nq, v, prep)
            rows.append(dict(exp="E2", name=f"qft6-{label}-x{x}", qc=qc, readout=[(homes, None)], repeats=2))
    for h in TFIM_FIELDS:
        ops = logical_tfim(N_LOGICAL, TROTTER_STEPS, h)
        variants = [("monolithic", 2, "monolithic")] + [(f"{nq}qpu-{v}", nq, v) for nq in LAYOUTS
                                                      for v in ("static", "migrate")]
        for label, nq, v in variants:
            qc, homes = distributed(ops, LAYOUTS[nq], nq, v)
            rows.append(dict(exp="E2", name=f"tfim6-{label}-h{h}", qc=qc, readout=[(homes, None)], repeats=2))
    return rows


def plan():
    """All 52 planned circuits: Experiment 1 runs 3 times (drift), Experiment 2 twice."""
    return experiment1() + experiment2()


def simulator_set():
    """Representative circuits that fit the 29-qubit simulator limit."""
    tele = [(i, b, "I") for (i, b) in TELEPORT_SETTINGS][:9]
    rcx = [(i, b, "I") for (i, b) in CNOT_SETTINGS][:7]
    (qt, rt), = packed("teleport", tele, max_qubits=27)
    (qr, rr), = packed("rcnot", rcx, max_qubits=28)
    rows = [dict(name="sim-teleport-pack9", qc=qt, readout=rt),
            dict(name="sim-remote-cnot-pack7", qc=qr, readout=rr)]
    e2 = {r["name"]: r for r in experiment2()}
    x0 = sorted(qft_inputs(), key=str)[0]
    for v in ["monolithic", "2qpu-static", "2qpu-migrate"]:
        rows.append(e2[f"qft6-{v}-x{x0}"])
    for v in ["monolithic", "2qpu-static", "2qpu-migrate"]:
        rows.append(e2[f"tfim6-{v}-h1.0"])
    return rows


# ----------------------------------------------------------------------------- verification
def verify_all():
    """Ideal-simulator checks of every construction."""
    for inp, basis in TELEPORT_SETTINGS:
        qc = QuantumCircuit(3); teleport_copy(qc, 0, 1, 2, inp, basis)
        ref = QuantumCircuit(1); prep_1q(ref, 0, inp); rotate_to_basis(ref, 0, basis)
        assert dist_close(ideal_marginal(qc, [2]), ideal_marginal(ref, [0]))
    for inp, basis in CNOT_SETTINGS:
        qc = QuantumCircuit(4); remote_cnot_copy(qc, 0, 1, 2, 3, inp, basis)
        ref = QuantumCircuit(2); local_cnot_copy(ref, 0, 1, inp, basis)
        assert dist_close(ideal_marginal(qc, [0, 3]), ideal_marginal(ref, [0, 1]))
    # Pauli-insertion mixture = depolarised (Werner) pair of fidelity F
    F = 0.9
    for inp, basis in TELEPORT_SETTINGS:
        mix = {}
        for p in PAULIS:
            qc = QuantumCircuit(3); teleport_copy(qc, 0, 1, 2, inp, basis, p)
            w = F if p == "I" else (1 - F) / 3
            for k, v in ideal_marginal(qc, [2]).items():
                mix[k] = mix.get(k, 0) + w * v
        ref = QuantumCircuit(1); prep_1q(ref, 0, inp); rotate_to_basis(ref, 0, basis)
        r, s = ideal_marginal(ref, [0]), (4 * F - 1) / 3
        assert dist_close(mix, {k: s * r.get(k, 0) + (1 - s) / 2 for k in "01"})
    # every distributed variant reproduces its monolithic reference
    groups = {}
    for r in experiment2():
        key = r["name"].split("-")[0] + "-" + r["name"].split("-")[-1]
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        dists = [ideal_marginal(r["qc"], r["readout"][0][0]) for r in rs]
        assert all(dist_close(d, dists[0], 1e-7) for d in dists), key
        if key.startswith("qft6"):
            assert max(dists[0].values()) > 1 - 1e-7
