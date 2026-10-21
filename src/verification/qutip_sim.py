import numpy as np
import qutip as qt
from src.config import ExperimentConfig

def verify_pulse_qutip(config: ExperimentConfig, I_pulse: np.ndarray, Q_pulse: np.ndarray, c_ops: list = None):
    """
    Simulates the JAX-optimized pulse in QuTiP to verify fidelity and test against noise.
    """
    if c_ops is None:
        c_ops = []

    # 1. Physical Constants
    alpha = 2 * np.pi * (config.hardware.anharmonicity_mhz / 1000.0)
    levels = config.hardware.levels
    
    # 2. QuTiP Operators
    a = qt.destroy(levels)
    
    # Drift Hamiltonian (Energy levels in rotating frame)
    energies = [-0.5 * alpha * n * (n - 1) for n in range(levels)]
    H0 = qt.Qobj(np.diag(energies))
    
    # Control Hamiltonians
    Hx = (a + a.dag()) / 2.0
    Hy = (a - a.dag()) / (2.0j)
    
    # 3. Time-dependent Hamiltonian format for QuTiP
    # Note: QuTiP expects cubic spline interpolation or raw arrays matching tlist
    H = [H0, [Hx, np.array(I_pulse)], [Hy, np.array(Q_pulse)]]
    
    # 4. Time array (converting dt to match the JAX steps)
    tlist = np.arange(0, config.pulse.duration_ns, config.pulse.dt_ns)
    
    # Ensure time list matches pulse length perfectly
    if len(tlist) > len(I_pulse):
        tlist = tlist[:len(I_pulse)]
    
    # 5. Run simulation
    psi0 = qt.basis(levels, 0) # Start in |0>
    target_state = qt.basis(levels, 1) # Target is |1> (for a pi-pulse)
    
    # mesolve handles both unitary evolution (no c_ops) and open systems (with c_ops)
    result = qt.mesolve(H, psi0, tlist, c_ops=c_ops)
    final_state = result.states[-1]
    
    # Calculate fidelity (accounting for density matrices if c_ops are used)
    if final_state.isket:
        fidelity = np.abs(final_state.overlap(target_state))**2
    else:
        fidelity = qt.fidelity(final_state, target_state)**2
        
    # Calculate leakage into |2>
    state_2 = qt.basis(levels, 2)
    if final_state.isket:
        leakage = np.abs(final_state.overlap(state_2))**2
    else:
        leakage = qt.expect(state_2 * state_2.dag(), final_state)
        
    return fidelity, leakage, result