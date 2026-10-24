import jax
import jax.numpy as jnp
import numpy as np
from src.physics.propagators import propagate, propagate_flux, propagate_parallel
from src.control.pulses import generate_analytic_pulse
from src.control.virtual_z import build_virtual_z


def build_local_drive_operators(levels=3):
    """Builds 9x9 two-qubit drive operators for single-qubit microwave control."""
    # Fixed: complex128 to match jax_enable_x64 and prevent 1e-8 truncation
    a = jnp.diag(jnp.sqrt(jnp.arange(1, levels)), k=1).astype(jnp.complex128)
    
    Hx = (a + a.conj().T) / 2.0
    Hy = (a - a.conj().T) / (2.0j)
    eye = jnp.eye(levels, dtype=jnp.complex128)
    
    Hx1 = jnp.kron(Hx, eye)
    Hy1 = jnp.kron(Hy, eye)
    
    Hx2 = jnp.kron(eye, Hx)
    Hy2 = jnp.kron(eye, Hy)
    
    return Hx1, Hy1, Hx2, Hy2


def build_bell_waveforms(duration_ns, dt, cal_a, cal_d, cal_pulse_rad, v_phase2=0.0):
    """
    Shared waveform builder for both the JAX Digital Twin and the QuTiP Fridge.
    If v_phase2 is provided, shifts the U3 microwave drive frame physically:
    Z(v_phase2) before U3 is equivalent to rotating the U3 drive quadratures by -v_phase2
    plus a final frame rotation Z(v_phase2) after U3.
    """
    I_half, Q_half = generate_analytic_pulse(
        duration_ns=duration_ns, 
        dt_ns=dt, 
        anharmonicity_mhz=cal_a, 
        pulse_type='drag', 
        target_angle=np.pi / 2.0
    )
    I_env = jnp.array(I_half, dtype=jnp.float64)
    Q_drag = jnp.array(Q_half, dtype=jnp.float64)
    cz_flux = jnp.array(cal_pulse_rad, dtype=jnp.float64)

    t = jnp.arange(I_env.shape[0]) * dt
    detuning_rad = 2.0 * jnp.pi * (cal_d / 1000.0)

    # Prep stage drives: Q1 modulated at detuning_rad, Q2 on resonance
    I1_prep = Q_drag * jnp.cos(detuning_rad * t) + I_env * jnp.sin(detuning_rad * t)
    Q1_prep = I_env * jnp.cos(detuning_rad * t) - Q_drag * jnp.sin(detuning_rad * t)
    I2_prep = Q_drag
    Q2_prep = I_env

    # U3 stage drives on Q2: nominal (-Q_drag, -I_env), optionally rotated by v_phase2
    I2_u3_nom = -Q_drag
    Q2_u3_nom = -I_env
    cos_p = jnp.cos(v_phase2)
    sin_p = jnp.sin(v_phase2)
    I2_u3 = I2_u3_nom * cos_p - Q2_u3_nom * sin_p
    Q2_u3 = I2_u3_nom * sin_p + Q2_u3_nom * cos_p

    return {
        "I_env": I_env,
        "Q_drag": Q_drag,
        "cz_flux": cz_flux,
        "I1_prep": I1_prep,
        "Q1_prep": Q1_prep,
        "I2_prep": I2_prep,
        "Q2_prep": Q2_prep,
        "I2_u3": I2_u3,
        "Q2_u3": Q2_u3,
    }


@jax.jit
def _compute_stage_propagators(
    I_env, Q_drag, cz_flux_pulse, dt, H0_static, Hx1, Hy1, Hx2, Hy2, Hz_control, detuning_mhz
):
    """Computes the three stage unitaries (U_prep, U_cz, U3) a single time."""
    t = jnp.arange(I_env.shape[0]) * dt
    detuning_rad = 2.0 * jnp.pi * (detuning_mhz / 1000.0)
    
    I_mod_X = Q_drag * jnp.cos(detuning_rad * t) + I_env * jnp.sin(detuning_rad * t)
    I_mod_Y = I_env * jnp.cos(detuning_rad * t) - Q_drag * jnp.sin(detuning_rad * t)
    
    U_prep = propagate_parallel(
        I_mod_X, I_mod_Y, Q_drag, I_env, 
        dt, H0_static, Hx1, Hy1, Hx2, Hy2
    )
    U_cz = propagate_flux(cz_flux_pulse, dt, H0_static, Hz_control)
    U3 = propagate(-Q_drag, -I_env, dt, H0_static, Hx2, Hy2)
    
    return U_prep, U_cz, U3


@jax.jit
def _compute_prep_and_cz(
    I_env, Q_drag, cz_flux_pulse, dt, H0_static, Hx1, Hy1, Hx2, Hy2, Hz_control, detuning_mhz
):
    """Computes the v_phase-independent U_prep and U_cz unitaries once."""
    t = jnp.arange(I_env.shape[0]) * dt
    detuning_rad = 2.0 * jnp.pi * (detuning_mhz / 1000.0)
    
    I_mod_X = Q_drag * jnp.cos(detuning_rad * t) + I_env * jnp.sin(detuning_rad * t)
    I_mod_Y = I_env * jnp.cos(detuning_rad * t) - Q_drag * jnp.sin(detuning_rad * t)
    
    U_prep = propagate_parallel(
        I_mod_X, I_mod_Y, Q_drag, I_env, 
        dt, H0_static, Hx1, Hy1, Hx2, Hy2
    )
    U_cz = propagate_flux(cz_flux_pulse, dt, H0_static, Hz_control)
    return U_prep, U_cz


@jax.jit
def _compute_u3_with_phase(I_env, Q_drag, dt, H0_static, Hx2, Hy2, v_phase2):
    """Computes U3 with Q2's microwave drive frame physically rotated by v_phase2."""
    cos_p = jnp.cos(v_phase2)
    sin_p = jnp.sin(v_phase2)
    I2_u3 = -Q_drag * cos_p + I_env * sin_p
    Q2_u3 = -Q_drag * sin_p - I_env * cos_p
    return propagate(I2_u3, Q2_u3, dt, H0_static, Hx2, Hy2)


@jax.jit
def simulate_bell_state_sequence(
    I_env, Q_drag, cz_flux_pulse, dt, H0_static, Hx1, Hy1, Hx2, Hy2, Hz_control, detuning_mhz,
    v_phase1=0.0, v_phase2=0.0
):
    U_prep, U_cz = _compute_prep_and_cz(
        I_env, Q_drag, cz_flux_pulse, dt, H0_static, Hx1, Hy1, Hx2, Hy2, Hz_control, detuning_mhz
    )
    U_mid = U_cz @ U_prep
    U3 = _compute_u3_with_phase(I_env, Q_drag, dt, H0_static, Hx2, Hy2, v_phase2)
    
    # Apply Virtual-Z frame rotation at the end (matching hardware apply_frame)
    Z = jnp.kron(build_virtual_z(v_phase1, 3), build_virtual_z(v_phase2, 3))
    U_bell = Z @ (U3 @ U_mid)
    return U_bell, U_mid


def generate_bell_state(
    duration_ns, dt, cal_a, cal_d, cal_pulse_rad, 
    H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control,
    v_phase1=0.0, v_phase2=0.0
):
    """
    Executes the Bell state sequence in a single shot on the noiseless JAX Digital Twin
    using pre-calibrated Virtual-Z phases.
    """
    wf = build_bell_waveforms(duration_ns, dt, cal_a, cal_d, cal_pulse_rad)
    psi_00 = jnp.zeros((9,), dtype=jnp.complex128).at[0].set(1.0)
    
    U_bell, _ = simulate_bell_state_sequence(
        wf["I_env"], wf["Q_drag"], wf["cz_flux"], 
        dt, H0_cal, Hx1, Hy1, Hx2, Hy2, Hz_control, cal_d, 
        v_phase1=v_phase1, v_phase2=v_phase2
    )
    
    psi_final = U_bell @ psi_00
    idx_comp = np.array([0, 1, 3, 4])
    rho_9x9 = np.outer(psi_final, jnp.conj(psi_final))
    rho_corr = rho_9x9[np.ix_(idx_comp, idx_comp)]
    
    return rho_corr

def pack_sequence(wf):
    """Concatenate prep | CZ | U3 into one timeline for HiddenDevice.play_sequence."""
    n, m = len(wf["I_env"]), len(wf["cz_flux"])
    z = lambda k: np.zeros(k)
    a = lambda k: np.asarray(wf[k])
    return {
        "I1": np.r_[a("I1_prep"), z(m), z(n)], "Q1": np.r_[a("Q1_prep"), z(m), z(n)],
        "I2": np.r_[a("I2_prep"), z(m), a("I2_u3")], "Q2": np.r_[a("Q2_prep"), z(m), a("Q2_u3")],
        "flux": np.r_[z(n), a("cz_flux"), z(n)],
    }
