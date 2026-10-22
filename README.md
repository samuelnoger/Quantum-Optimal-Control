# Superconducting Two-Qubit CZ Gate: Calibrate-Then-Control Pipeline

This repository contains a high-fidelity simulation and optimization project for superconducting transmon qutrits. The goal is to build a complete calibration and control pipeline that first characterizes an unknown hardware device using a vectorized Swap Chevron fitter, synthesizes a leakage-suppressed, bandwidth-limited Controlled-Z (CZ) gate, and integrates the calibrated pulses into an end-to-end Bell state ($\Phi^+$) circuit.

The pipeline prioritizes simulation accuracy using a high-performance, JAX-based digital twin and robustly validates performance against a QuTiP-based `HiddenDevice` hardware simulator incorporating $T_1/T_2$ decoherence and projective measurement shot noise.

## Methods & Implementation

**1. The Digital Twin & Calibration Fitter**
Instead of relying on nominal design guesses, the pipeline calibrates unknown hardware parameters simultaneously:

- **Vectorized Swap Chevron Fitter:** Fits exchange coupling ($g$), detuning ($\Delta$), and transmon anharmonicity ($\alpha$) concurrently by probing both $\vert{}10\rangle$ and $\vert{}11\rangle$ initial states.
- **JAX-Jitted Propagators:** Uses fast analytical matrix exponentials (`expm`) per pulse segment to accelerate gradient-based optimization via Optax (`Adam`).

**2. Bandwidth-Limited CZ Gate Optimization**

- **Fourier-Basis Pulse Shaping:** Sculpts smooth flux pulses using sine harmonics filtered through a Gaussian kernel to respect real hardware AWG bandwidth limits and prevent high-frequency leakage.
- **Leakage Suppression:** Optimizes against non-computational qutrit states (such as $\vert{}20\rangle$) to achieve high gate fidelity.

**3. Verification & Benchmarking Infrastructure**

- **Held-Out Cross-Validation:** Tests the calibrated digital twin against 200 random, time-varying waveforms it has never seen before to prove generalization accuracy.
- **Conditional Ramsey Experiments:** Simulates physical phase-fringes on the QuTiP hardware simulator to verify the exact $\pi$ ($180^\circ$) conditional phase shift of the CZ gate.

**4. End-to-End Bell State Synthesis & Virtual-Z Calibration**

- **Full-Circuit Pulse Integration:** Combines parallel single-qubit DRAG preparation pulses (with detuning modulation), the flux-tuned CZ gate, and a final single-qubit rotation in the full 9D two-qutrit space.
- **Calibrate-Once Virtual-Z Sweep:** Calibrates single-qubit Virtual-Z phase corrections on the Digital Twin using a precomputed 2D grid search and Nelder-Mead refinement to compensate for always-on coupling and local phase accumulation.
- **Physical Frame Execution on Hardware:** Freezes the twin-calibrated phases and executes the full concatenated waveform timeline (`play_sequence`) on the open-system QuTiP `HiddenDevice` by physically shifting the subsequent microwave drive phase and rotating the final tomography readout frame.

## Key Observations

- **Parameter Convergence:** The JAX calibration fitter converges to true hardware parameters within $<0.01\text{ MHz}$ error even under realistic finite-shot measurement noise ($2,000$ shots).
- **Generalization:** Cross-validation against held-out random waveforms showed that calibration reduces population prediction error from $\sim 9.3\%$ down to the physical $T_1/T_2$ noise floor ($\sim 0.09\%$).
- **Hardware Validation:** Comparing uncalibrated versus calibrated pulses on the QuTiP fridge proves that calibration is essential for eliminating population leakage and achieving an accurate $180^\circ$ phase flip.
- **End-to-End Entanglement:** Playing the full Bell state pulse sequence with frozen twin-calibrated Virtual-Z phases on the noisy QuTiP `HiddenDevice` achieves a **98.10%** $\Phi^+$ Bell state fidelity while retaining **99.48%** population in the computational subspace ($0.52\%$ qutrit leakage).

## Notebooks Overview

- **`calibrate_and_control.ipynb`**: The main interactive calibration and validation workflow. Executes the Digital Twin Swap Chevron calibration, trains the CZ pulse, and cross-validates the calibrated model against the QuTiP `HiddenDevice` simulator.
- **`two_qubit_cz_gate.ipynb`**: Focuses on optimal CZ gate control. Uses JAX-based optimization to shape bandwidth-limited Fourier flux pulses and visualizes the state population and conditional phase dynamics across the $\vert{}11\rangle \leftrightarrow \vert{}20\rangle$ avoided crossing.
- **`analog_bell_state.ipynb`**: The capstone notebook. Performs a 2D Virtual-Z calibration sweep on the Digital Twin, freezes the optimal phase offsets, and executes the complete DRAG + CZ pulse sequence on both the noiseless JAX twin and the open-system QuTiP hardware simulator to generate a 98.10% fidelity $\Phi^+$ Bell state.

## Project Structure

```text
configs/two_qubit_CZ.yaml               Hardware and optimization hyperparameter configuration
src/config.py                           Configuration parser
src/physics/operators.py                Transmon and two-qubit Hamiltonian builders
src/physics/propagators.py              JAX time-evolution and matrix exponentiation step functions
src/control/pulses.py                   DRAG pulse generator, Fourier basis, and Gaussian filters
src/control/sequences.py                Bell state pulse sequences and 2D Virtual-Z calibration
src/optimization/losses.py              Loss functions for calibration and CZ gate optimization
src/optimization/fitters.py             Digital twin calibration fitter (DigitalTwinFitter)
src/optimization/training_loops.py      Optimization loop and setup helpers
src/verification/qutip_device.py        QuTiP 'HiddenDevice' hardware simulator with decoherence & shot noise
src/verification/benchmarks.py          Cross-validation and Conditional Ramsey experiment runners
src/verification/plots.py               3D density matrix cityscape visualization helpers
notebooks/two_qubit_cz_gate.ipynb       Optimal CZ control and state dynamics
notebooks/calibrate_and_control.ipynb   Main interactive calibration and validation workflow
notebooks/analog_bell_state.ipynb       Full pipeline integration and 98.10% Bell state generation

```

## Acknowledgements

I built this project to deepen my hands-on experience with superconducting quantum hardware simulation, digital twin calibration, and optimal pulse control. As a personal learning and development project, much of the codebase, mathematical structuring, and debugging was developed with the assistance of AI coding tools.