"""Compile circuits to IonQ native gates (GPi, GPi2, ZZ) for native-gateset submission.

Method:
  1. Qiskit transpile to {rz, sx, x, rzz} at optimization level 3. Every CNOT, CZ, CP and RZZ becomes
     one RZZ plus single-qubit gates, and each single-qubit block uses at most two sx.
  2. rz is a virtual frame change. It commutes with ZZ and is absorbed into the phase of the next
     GPi/GPi2 on that qubit, using Rz(-l) RX(pi/2) Rz(l) = GPi2(-l / 2pi) up to a global phase:
         sx -> GPi2(-f / 2pi),   x -> GPi(-f / 2pi),   rzz(phi) -> ZZ(phi / 2pi)     (phases in turns)
     where f is the accumulated rz angle on that qubit. Frames left before the final Z-basis
     measurement do not change the measurement statistics.
  3. `check()` confirms that the native circuit reproduces the original output distribution.
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ionq import GPI2Gate, GPIGate, ZZGate

import circuits as C

BASIS = ["rz", "sx", "x", "rzz"]


def to_native(qc: QuantumCircuit) -> QuantumCircuit:
    src = qc.remove_final_measurements(inplace=False)
    t = transpile(src, basis_gates=BASIS, optimization_level=3, seed_transpiler=11)
    out = QuantumCircuit(src.num_qubits)
    frame = [0.0] * src.num_qubits
    for inst in t.data:
        name = inst.operation.name
        qs = [t.find_bit(q).index for q in inst.qubits]
        if name == "rz":
            frame[qs[0]] += float(inst.operation.params[0])
        elif name == "sx":
            out.append(GPI2Gate((-frame[qs[0]] / (2 * math.pi)) % 1.0), qs)
        elif name == "x":
            out.append(GPIGate((-frame[qs[0]] / (2 * math.pi)) % 1.0), qs)
        elif name == "rzz":
            theta = (float(inst.operation.params[0]) / (2 * math.pi) + 0.5) % 1.0 - 0.5
            if abs(theta) > 1e-12:
                out.append(ZZGate(theta), qs)
        elif name not in ("barrier", "delay", "global_phase"):
            raise ValueError(f"unexpected gate {name}")
    out.measure_all()
    return out


def gate_counts(native: QuantumCircuit):
    ops = native.count_ops()
    return int(ops.get("gpi", 0) + ops.get("gpi2", 0)), int(ops.get("zz", 0))


def check(qc: QuantumCircuit, native: QuantumCircuit, groups, tol=1e-6):
    """The native circuit gives the same ideal distribution on every readout group."""
    n2 = transpile(native.remove_final_measurements(inplace=False),
                   basis_gates=["u", "cx", "rzz"], optimization_level=0)
    for g in groups:
        assert C.dist_close(C.ideal_marginal(qc, g), C.ideal_marginal(n2, g), tol), g


def self_test(trials=30):
    rng = np.random.default_rng(0)
    for _ in range(trials):
        n = int(rng.integers(2, 6))
        qc = QuantumCircuit(n)
        for _ in range(25):
            k = rng.integers(0, 7)
            a, b = (int(v) for v in rng.choice(n, 2, replace=False))
            th = float(rng.uniform(-math.pi, math.pi))
            [lambda: qc.h(a), lambda: qc.rx(th, a), lambda: qc.ry(th, a), lambda: qc.rz(th, a),
             lambda: qc.cx(a, b), lambda: qc.cz(a, b), lambda: qc.cp(th, a, b)][k]()
        p = Statevector.from_instruction(qc).probabilities()
        q = Statevector.from_instruction(to_native(qc).remove_final_measurements(inplace=False)).probabilities()
        assert np.allclose(p, q, atol=1e-8)
