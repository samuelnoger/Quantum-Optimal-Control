# src/optimization/losses.py
import jax
import jax.numpy as jnp
from src.physics.propagators import propagate, propagate_flux
from src.control.pulses import apply_low_pass_filter
from src.physics.operators import assemble_H0

@jax.jit
def compute_loss(params, dt, H0, Hx, Hy, U_target, basis, penalty_weight=0.1, reg_weight=1e-4):
    cI, cQ = params['I_coeffs'], params['Q_coeffs']
    I_pulse, Q_pulse = basis @ cI, basis @ cQ
    U_final = propagate(I_pulse, Q_pulse, dt, H0, Hx, Hy)

    d = U_target.shape[0]
    U_comp = U_final[:d, :d]
    tr = jnp.trace(U_target.conj().T @ U_comp)
    
    # Do not use jnp.abs() here because its gradient is NaN exactly at 0
    infidelity = 1.0 - (tr.real ** 2 + tr.imag ** 2) / d ** 2          
    
    leak_block = U_final[d:, :d]
    leakage = jnp.sum(leak_block.real ** 2 + leak_block.imag ** 2)

    # Regularization: Penalize mean power and heavily penalize high harmonics
    n = jnp.arange(1, cI.shape[0] + 1)
    reg = reg_weight * (jnp.mean(I_pulse ** 2 + Q_pulse ** 2) + 0.05 * jnp.sum(n ** 2 * (cI ** 2 + cQ ** 2)) / I_pulse.shape[0])

    loss = infidelity + penalty_weight * leakage + reg
    return loss, {"infidelity": infidelity, "leakage": leakage}

@jax.jit
def compute_cz_loss(params, dt, H0, Hz_control, basis, kernel, f_max, penalty_weight=0.5):
    """params = {'flux_coeffs': (K,), 'phases': (2,)}.   f_max in rad/ns (= 2 pi * GHz), basis = sin(k pi t / T)."""
    raw = basis @ params['flux_coeffs']
    flux = f_max * jnp.tanh(raw / f_max)                       # |flux| <= f_max; units rad/ns, as propagate_flux expects
    flux = apply_low_pass_filter(flux, kernel)
 
    U_final = propagate_flux(flux, dt, H0, Hz_control)
    idx = jnp.array([0, 1, 3, 4])                              # |00>, |01>, |10>, |11> in the 3x3-level product space
    U_comp = U_final[idx[:, None], idx]
 
    # CZ up to single-qubit Z phases (virtual Z gates): target = diag(1, e^{i p2}, e^{i p1}, -e^{i (p1 + p2)})
    p1, p2 = params['phases'][0], params['phases'][1]
    phase = jnp.exp(1j * jnp.stack([jnp.zeros_like(p1), p2, p1, p1 + p2]))
    target_diag = jnp.array([1.0, 1.0, 1.0, -1.0]) * phase
    tr = jnp.sum(target_diag.conj() * jnp.diag(U_comp))
    infidelity = 1.0 - (tr.real ** 2 + tr.imag ** 2) / 16.0
 
    leakage = jnp.mean(1.0 - jnp.sum(U_comp.real ** 2 + U_comp.imag ** 2, axis=0))   # no abs(): NaN gradient at 0
    return infidelity + penalty_weight * leakage, {"infidelity": infidelity, "leakage": leakage}


@jax.jit
def calibration_loss(calibration_params, dt_step, H_anharm_base, H_int_base, Hz_control, amplitudes_rad, target_pops_grid_10, target_pops_grid_11):
    c_mhz = calibration_params['coupling_mhz']
    d_mhz = calibration_params['detuning_mhz']
    alpha_mhz = calibration_params['anharmonicity_mhz']
    
    # Rebuild base operators with guessed alpha if it's dynamic
    # (Or pass modified H_anharm_base if built dynamically)
    H0_guess = assemble_H0(H_anharm_base, H_int_base, Hz_control, c_mhz, d_mhz, alpha_mhz)
    
    psi_init_10 = jnp.zeros(9, dtype=jnp.complex128).at[3].set(1.0) # |10>
    psi_init_11 = jnp.zeros(9, dtype=jnp.complex128).at[4].set(1.0) # |11>
    idx = jnp.array([0, 1, 3, 4])
    num_durs = target_pops_grid_10.shape[1]

    def eval_single_amplitude(amp_rad, targets_10, targets_11):
        H_total = H0_guess + amp_rad * Hz_control
        U_step = jax.scipy.linalg.expm(-1j * H_total * dt_step)

        # Simulate |10> trajectory
        def scan_10(psi, _):
            psi_next = U_step @ psi
            pops = psi_next.real ** 2 + psi_next.imag ** 2
            return psi_next, pops[idx]
        _, pred_10 = jax.lax.scan(scan_10, psi_init_10, None, length=num_durs)

        # Simulate |11> trajectory
        def scan_11(psi, _):
            psi_next = U_step @ psi
            pops = psi_next.real ** 2 + psi_next.imag ** 2
            return psi_next, pops[idx]
        _, pred_11 = jax.lax.scan(scan_11, psi_init_11, None, length=num_durs)

        loss_10 = jnp.mean((pred_10 - targets_10) ** 2)
        loss_11 = jnp.mean((pred_11 - targets_11) ** 2)
        return loss_10 + loss_11

    losses = jax.vmap(eval_single_amplitude, in_axes=(0, 0, 0))(amplitudes_rad, target_pops_grid_10, target_pops_grid_11)
    return jnp.mean(losses)