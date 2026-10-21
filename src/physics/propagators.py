import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

@jax.jit
def propagate(I_pulse, Q_pulse, dt, H0, Hx, Hy):
    """
    Simulates the time evolution. Works for any Hilbert space dimension.
    """
    def step(U, controls):
        I, Q = controls
        H = H0 + I * Hx + Q * Hy
        U_step = expm(-1j * H * dt)
        return U_step @ U, None 

    # Dynamically size the identity matrix based on the Hamiltonian
    dim = H0.shape[0]
    U_init = jnp.eye(dim, dtype=jnp.complex128)
    
    U_final, _ = jax.lax.scan(step, U_init, (I_pulse, Q_pulse))
    return U_final

@jax.jit
def propagate_parallel(I1, Q1, I2, Q2, dt, H0_static, Hx1, Hy1, Hx2, Hy2):
    """
    Simulates simultaneous time evolution under parallel drives on both channels.
    H(t) = H0_static + I1(t)*Hx1 + Q1(t)*Hy1 + I2(t)*Hx2 + Q2(t)*Hy2
    """
    def step(U, controls):
        i1, q1, i2, q2 = controls
        H = H0_static + i1 * Hx1 + q1 * Hy1 + i2 * Hx2 + q2 * Hy2
        U_step = expm(-1j * H * dt)
        return U_step @ U, None 

    dim = H0_static.shape[0]
    U_init = jnp.eye(dim, dtype=jnp.complex128)
    
    # Scan over all four control streams simultaneously time-step by time-step
    U_final, _ = jax.lax.scan(step, U_init, (I1, Q1, I2, Q2))
    return U_final


@jax.jit
def propagate_flux(flux_pulse, dt, H0, Hz_control):
    """
    Simulates time evolution for a Z-controlled (flux) gate.
    Dynamically scales to the 9-dimensional two-qubit space.
    """
    def step(U, flux_val):
        # H0 contains the static drift and constant exchange coupling.
        # flux_val * Hz_control shifts the target qubit's frequency.
        H = H0 + flux_val * Hz_control
        U_step = expm(-1j * H * dt)
        return U_step @ U, None 

    dim = H0.shape[0]
    U_init = jnp.eye(dim, dtype=jnp.complex128)
    
    # We only scan over the 1D flux_pulse array
    U_final, _ = jax.lax.scan(step, U_init, flux_pulse)
    return U_final
