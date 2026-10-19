# src/physics/propagators.py
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
