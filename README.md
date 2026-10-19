# Superconducting Two-Qubit CZ Gate: Calibrate-Then-Control Pipeline

This repository contains a high-fidelity simulation and optimization project for superconducting transmon qubits. The goal is to build a complete calibration and control pipeline that first characterizes an unknown hardware device using a vectorized Swap Chevron fitter and then synthesizes a leakage-suppressed, bandwidth-limited Controlled-Z (CZ) gate.

The pipeline prioritizes simulation accuracy using a high-performance, JAX-based workflow and robustly validates performance against a QuTiP-based "HiddenDevice" hardware simulator incorporating $T_1/T_2$ decoherence and projective measurement shot noise.

## Methods & Implementation

**1. The Digital Twin & Calibration Fitter**
Instead of relying on nominal design guesses, the pipeline calibrates unknown hardware parameters simultaneously:

* **Vectorized Swap Chevron Fitter:** Fits exchange coupling ($g$), detuning ($\Delta$), and transmon anharmonicity ($\alpha$) concurrently by probing both $|10\rangle$ and $|11\rangle$ initial states.
* **JAX-Jitted Propagators:** Uses fast analytical matrix exponentials (`expm`) per pulse segment to accelerate gradient-based optimization via Optax (`Adam`).

**2. Bandwidth-Limited CZ Gate Optimization**

* **Fourier-Basis Pulse Shaping:** Sculpting smooth flux pulses using sine harmonics filtered through a Gaussian kernel to respect real hardware bandwidth limits and prevent high-frequency leakage.
* **Leakage Suppression:** Optimizing against non-computational states (such as $|20\rangle$) to achieve high gate fidelity.

**3. Verification & Benchmarking Infrastructure**

* **Held-Out Cross-Validation:** Tests the calibrated digital twin against completely random, time-varying waveforms it has never seen before to prove prediction accuracy.
* **Conditional Ramsey Experiments:** Simulates physical phase-fringes on the QuTiP hardware simulator to verify the exact $\pi$ ($180^\circ$) conditional phase shift of the CZ gate.

## Key Observations

* **Parameter Convergence:** The JAX calibration fitter converges to true hardware parameters within $<0.01\text{ MHz}$ error even under realistic finite-shot measurement noise ($2,000$ shots).
* **Generalization:** Cross-validation against held-out random waveforms showed that calibration reduces population prediction error from $\sim 9.3\%$ down to the physical $T_1/T_2$ noise floor ($\sim 0.09\%$).
* **Hardware Validation:** Comparing uncalibrated versus calibrated pulses on the QuTiP fridge proves that calibration is essential for eliminating population leakage and achieving an accurate $180^\circ$ phase flip.

## Project Structure

```text
configs/two_qubit_CZ.yaml               Hardware and optimization hyperparameter configuration
src/config.py                           Configuration parser
src/physics/operators.py                Transmon and two-qubit Hamiltonian builders
src/physics/propagators.py              JAX time-evolution and matrix exponentiation step functions
src/control/pulses.py                   Fourier basis generator and Gaussian low-pass filters
src/optimization/losses.py              Loss functions for calibration and CZ gate optimization
src/optimization/fitters.py             Digital twin calibration fitter (DigitalTwinFitter)
src/optimization/training_loops.py      Optimization loop and setup helpers
src/verification/qutip_device.py        QuTiP 'HiddenDevice' hardware simulator with decoherence & shot noise
src/verification/benchmarks.py          Cross-validation and Conditional Ramsey experiment runners
notebooks/two_qubit_cz_gate.ipynb       Optimal CZ control
notebooks/calibrate_and_control.ipynb   Main interactive workflow notebook

```

## Acknowledgements

I built this project to deepen my hands-on experience with superconducting quantum hardware simulation, digital twin calibration, and optimal pulse control. As a personal learning and development project, much of the codebase, mathematical structuring, and debugging was developed with the assistance of AI coding tools.