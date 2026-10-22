import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from src.physics.propagators import propagate, propagate_flux, propagate_parallel
from src.control.pulses import generate_analytic_pulse


def build_local_drive_operators(levels=3):
    """Builds 9x9 two-qubit drive operators for single-qubit microwave control."""
    a = jnp.diag(jnp.sqrt(jnp.arange(1, levels)), k=1).astype(jnp.complex64)
    
    Hx = (a + a.conj().T) / 2
    Hy = (a - a.conj().T) / (2j)
    eye = jnp.eye(levels)
    
    Hx1 = jnp.kron(Hx, eye)
    Hy1 = jnp.kron(Hy, eye)
    
    Hx2 = jnp.kron(eye, Hx)
    Hy2 = jnp.kron(eye, Hy)
    
    return Hx1, Hy1, Hx2, Hy2


def build_virtual_z(phase_rad, levels=3):
    """Builds a single-qubit Z-rotation matrix."""
    return jnp.diag(jnp.exp(-1j * phase_rad * jnp.arange(levels)))


@jax.jit
def simulate_bell_state_sequence(
    I_env, Q_drag, cz_flux_pulse, dt, H0_static, Hx1, Hy1, Hx2, Hy2, Hz_control, detuning_mhz,
    v_phase1=0.0, v_phase2=0.0
):
    t = jnp.arange(I_env.shape[0]) * dt
    detuning_rad = 2 * jnp.pi * (detuning_mhz / 1000.0)
    
    # Modulate Qubit 1 drives with frame detuning
    I_mod_X = Q_drag * jnp.cos(detuning_rad * t) + I_env * jnp.sin(detuning_rad * t)
    I_mod_Y = I_env * jnp.cos(detuning_rad * t) - Q_drag * jnp.sin(detuning_rad * t)
    
    # Parallel single-qubit preparation
    U_prep = propagate_parallel(
        I_mod_X, I_mod_Y, Q_drag, I_env, 
        dt, H0_static, Hx1, Hy1, Hx2, Hy2
    )
    
    # Flux-tuned CZ gate
    U_cz = propagate_flux(cz_flux_pulse, dt, H0_static, Hz_control)
    
    # Apply Virtual-Z phase frame corrections
    Z1 = jnp.kron(build_virtual_z(v_phase1, 3), jnp.eye(3))
    Z2 = jnp.kron(jnp.eye(3), build_virtual_z(v_phase2, 3))
    U_mid = Z2 @ Z1 @ U_cz @ U_prep
    
    # Final single-qubit rotation on Q2: Ry(-pi/2)
    U3 = propagate(-Q_drag, -I_env, dt, H0_static, Hx2, Hy2)
    
    U_bell = U3 @ U_mid
    return U_bell, U_mid


def calibrate_virtual_z(
    duration_ns, dt, cal_a, cal_d, cal_pulse_rad, 
    H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control, n_grid=24
):
    """
    Calibrates Virtual-Z phases once for a given pulse configuration by maximizing 
    target Bell state overlap (tomography fidelity) via a 2D grid sweep + Nelder-Mead.
    """
    I_half, Q_half = generate_analytic_pulse(
        duration_ns=duration_ns, 
        dt_ns=dt, 
        anharmonicity_mhz=cal_a, 
        pulse_type='drag', 
        target_angle=np.pi/2
    )
    
    I_env = jnp.array(I_half)
    Q_drag = jnp.array(Q_half)
    cz_flux = jnp.array(cal_pulse_rad)
    
    psi_00 = jnp.zeros((9,), dtype=jnp.complex128).at[0].set(1.0)
    target = (
        jnp.zeros((9,), dtype=jnp.complex128)
        .at[0].set(1.0 / np.sqrt(2.0))
        .at[4].set(1.0 / np.sqrt(2.0))
    )

    @jax.jit
    def eval_infidelity(phases):
        U_bell, _ = simulate_bell_state_sequence(
            I_env, Q_drag, cz_flux, dt, H0_cal, 
            Hx1, Hy1, Hx2, Hy2, Hz_control, cal_d,
            v_phase1=phases[0], v_phase2=phases[1]
        )
        psi_final = U_bell @ psi_00
        return 1.0 - jnp.abs(jnp.vdot(target, psi_final)) ** 2

    # Fast vectorized 2D coarse grid search
    grid = jnp.linspace(-jnp.pi, jnp.pi, n_grid, endpoint=False)
    g1, g2 = jnp.meshgrid(grid, grid, indexing="ij")
    grid_points = jnp.stack([g1.ravel(), g2.ravel()], axis=-1)
    
    infidelities = np.array(jax.vmap(eval_infidelity)(grid_points))
    best_idx = int(np.argmin(infidelities))
    best_init = np.array(grid_points[best_idx])

    # Local Nelder-Mead refinement
    res = minimize(
        lambda x: float(eval_infidelity(jnp.array(x))),
        x0=best_init,
        method="Nelder-Mead",
        options=dict(xatol=1e-4, fatol=1e-8)
    )
    
    # Wrap phases cleanly into [-pi, pi]
    v_phase1 = float((res.x[0] + np.pi) % (2 * np.pi) - np.pi)
    v_phase2 = float((res.x[1] + np.pi) % (2 * np.pi) - np.pi)
    fidelity_grid = 1.0 - infidelities.reshape(n_grid, n_grid)

    return {
        "v_phase1": v_phase1,
        "v_phase2": v_phase2,
        "infidelity": float(res.fun),
        "grid_rad": np.array(grid),
        "fidelity_grid": fidelity_grid,
    }


def generate_bell_state(
    duration_ns, dt, cal_a, cal_d, cal_pulse_rad, 
    H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control,
    v_phase1=0.0, v_phase2=0.0
):
    """
    Executes the Bell state sequence in a single shot using pre-calibrated Virtual-Z phases.
    """
    I_half, Q_half = generate_analytic_pulse(
        duration_ns=duration_ns, 
        dt_ns=dt, 
        anharmonicity_mhz=cal_a, 
        pulse_type='drag', 
        target_angle=np.pi/2
    )
    
    I_env = jnp.array(I_half)
    Q_drag = jnp.array(Q_half)
    cz_flux = jnp.array(cal_pulse_rad)
    
    psi_00 = jnp.zeros((9,), dtype=jnp.complex128).at[0].set(1.0)
    
    U_bell, _ = simulate_bell_state_sequence(
        I_env, Q_drag, cz_flux, 
        dt, H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control, cal_d, 
        v_phase1=v_phase1, v_phase2=v_phase2
    )
    
    psi_final = U_bell @ psi_00
    idx_comp = np.array([0, 1, 3, 4])
    rho_9x9 = np.outer(psi_final, jnp.conj(psi_final))
    rho_corr = rho_9x9[np.ix_(idx_comp, idx_comp)]
    
    return rho_corr