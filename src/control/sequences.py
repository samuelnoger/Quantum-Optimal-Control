import jax
import jax.numpy as jnp
import numpy as np
from src.physics.propagators import propagate, propagate_flux, propagate_parallel
from src.control.pulses import generate_analytic_pulse



def build_local_drive_operators(levels=3):
    """Builds 9x9 two-qubit drive operators for single-qubit microwave control."""
    # Truncated annihilation operator
    a = jnp.diag(jnp.sqrt(jnp.arange(1, levels)), k=1).astype(jnp.complex64)
    
    Hx = (a + a.conj().T) / 2
    Hy = (a - a.conj().T) / (2j)
    eye = jnp.eye(levels)
    
    # Qubit 1 operators (kron with identity on Qubit 2)
    Hx1 = jnp.kron(Hx, eye)
    Hy1 = jnp.kron(Hy, eye)
    
    # Qubit 2 operators (identity on Qubit 1 kron with Qubit 2)
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
    
    # Q1 gets (I_mod_X, I_mod_Y) and Q2 gets (Q_drag, I_env) in the exact same time window
    U_prep = propagate_parallel(
        I_mod_X, I_mod_Y, Q_drag, I_env, 
        dt, H0_static, Hx1, Hy1, Hx2, Hy2
    )
    
    # Run the flux-tuned CZ gate
    U_cz = propagate_flux(cz_flux_pulse, dt, H0_static, Hz_control)
    
    # Track and apply virtual Z phase frame corrections
    Z1 = jnp.kron(build_virtual_z(v_phase1, 3), jnp.eye(3))
    Z2 = jnp.kron(jnp.eye(3), build_virtual_z(v_phase2, 3))
    U_mid = Z2 @ Z1 @ U_cz @ U_prep
    
    # Final decoupled single-qubit rotation on Q2: Ry(-pi/2)
    U3 = propagate(-Q_drag, -I_env, dt, H0_static, Hx2, Hy2)
    
    U_bell = U3 @ U_mid
    return U_bell, U_mid

def generate_bell_state(duration_ns, dt, cal_a, cal_d, cal_pulse_rad, 
                                     H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control):
    """
    Generates single-qubit physical pulses for a given duration, simulates 
    the raw sequence, auto-calibrates the Virtual Z phases, and returns 
    the final 4x4 density matrices.
    """
    # Generate the single-qubit microwave pulse
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
    
    # Define the 9D computational ground state |00>
    psi_00 = jnp.zeros((9,), dtype=jnp.complex128).at[0].set(1.0)
    
    # 1. Run raw sequence to measure Qubit 2's phase entering U3 (at U_mid)
    _, U_mid_raw = simulate_bell_state_sequence(
        I_env, Q_drag, cz_flux, 
        dt, H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control, cal_d, 
        v_phase1=0.0, v_phase2=0.0
    )
    psi_mid = U_mid_raw @ psi_00

    # Phase of Q2 when Q1 is in |0> (target relative phase = 0), and when Q1 is in |1> (target relative phase = pi)
    phi_q2_q1is0 = jnp.angle(psi_mid[1]) - jnp.angle(psi_mid[0])
    phi_q2_q1is1 = jnp.angle(psi_mid[4]) - jnp.angle(psi_mid[3]) - jnp.pi

    # True local Z2 frame is the symmetric average of both Q1 branches
    corr_2 = phi_q2_q1is0 + 0.5 * jnp.angle(jnp.exp(1j * (phi_q2_q1is1 - phi_q2_q1is0)))

    # 2. With Q2 aligned for U3, measure the total full-circuit phase on |11> (index 4) after U3
    U_bell_mid_cal, _ = simulate_bell_state_sequence(
        I_env, Q_drag, cz_flux, 
        dt, H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control, cal_d, 
        v_phase1=0.0, v_phase2=corr_2
    )
    psi_end = U_bell_mid_cal @ psi_00
    corr_1 = jnp.angle(psi_end[4]) - jnp.angle(psi_end[0])  # Tracks Q1 through the end of U3

    # 3. Run the sequence with both physical Virtual-Z phases applied
    U_bell_corr, _ = simulate_bell_state_sequence(
        I_env, Q_drag, cz_flux, 
        dt, H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control, cal_d, 
        v_phase1=corr_1, v_phase2=corr_2
    )
    
    # Extract computational 4x4 density matrices
    idx_comp = np.array([0, 1, 3, 4]) 
    rho_corr_9x9 = np.outer(U_bell_corr @ psi_00, jnp.conj(U_bell_corr @ psi_00))
    rho_corr = rho_corr_9x9[np.ix_(idx_comp, idx_comp)]
    
    return rho_corr, corr_1, corr_2