"""Virtual-Z matrices and tomography-frame operations."""

import numpy as np
import jax.numpy as jnp
import qutip as qt


def build_virtual_z(phase_rad, levels=3):
    """Build a single-qubit Z-rotation matrix."""
    return jnp.diag(jnp.exp(-1j * phase_rad * jnp.arange(levels)))


def apply_frame(rho, v1, v2, levels=3):
    """Apply the analytic tomography-frame rotation Z1(v1) x Z2(v2)."""
    zq = lambda phase: qt.Qobj(np.diag(np.exp(-1j * phase * np.arange(levels))))
    rotation = qt.tensor(zq(v1), zq(v2))
    if rho.dims != rotation.dims:
        expected_shape = (levels * levels, levels * levels)
        if rho.shape != expected_shape:
            raise ValueError(
                f"Expected a {expected_shape[0]}x{expected_shape[1]} two-qudit state, "
                f"got {rho.shape}."
            )
        rho = qt.Qobj(rho.full(), dims=rotation.dims)
    return rotation * rho * rotation.dag()
