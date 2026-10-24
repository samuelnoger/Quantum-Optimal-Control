"""Virtual-Z phase calibration on the digital twin."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from src.control.waveforms import (
    _compute_prep_and_cz,
    _compute_u3_with_phase,
    build_bell_waveforms,
)


def calibrate_virtual_z(
    duration_ns,
    dt,
    cal_a,
    cal_d,
    cal_pulse_rad,
    H0_cal,
    Hx1,
    Hy1,
    Hx2,
    Hy2,
    Hz_control,
    n_grid=48,
):
    """Calibrate virtual-Z phases using a grid search and local refinement."""
    wf = build_bell_waveforms(duration_ns, dt, cal_a, cal_d, cal_pulse_rad)
    U_prep, U_cz = _compute_prep_and_cz(
        wf["I_env"],
        wf["Q_drag"],
        wf["cz_flux"],
        dt,
        H0_cal,
        Hx1,
        Hy1,
        Hx2,
        Hy2,
        Hz_control,
        cal_d,
    )
    psi_00 = jnp.zeros((9,), dtype=jnp.complex128).at[0].set(1.0)
    psi_after_cz = (U_cz @ U_prep) @ psi_00
    target = (
        jnp.zeros((9,), dtype=jnp.complex128)
        .at[0].set(1.0 / np.sqrt(2.0))
        .at[4].set(1.0 / np.sqrt(2.0))
    )

    @jax.jit
    def psi_after_u3(v2):
        U3 = _compute_u3_with_phase(
            wf["I_env"], wf["Q_drag"], dt, H0_cal, Hx2, Hy2, v2
        )
        return U3 @ psi_after_cz

    @jax.jit
    def eval_infidelity(phases):
        v1, v2 = phases[0], phases[1]
        psi_u3 = psi_after_u3(v2)
        z1_diag = jnp.exp(-1j * v1 * jnp.arange(3))
        z2_diag = jnp.exp(-1j * v2 * jnp.arange(3))
        psi_final = jnp.kron(z1_diag, z2_diag) * psi_u3
        return 1.0 - jnp.abs(jnp.vdot(target, psi_final)) ** 2

    grid = jnp.linspace(-jnp.pi, jnp.pi, n_grid, endpoint=False)
    psi_u3_grid = jax.vmap(psi_after_u3)(grid)

    @jax.jit
    def eval_grid_point(i1, i2):
        v1, v2 = grid[i1], grid[i2]
        z_diag = jnp.kron(
            jnp.exp(-1j * v1 * jnp.arange(3)),
            jnp.exp(-1j * v2 * jnp.arange(3)),
        )
        psi_final = z_diag * psi_u3_grid[i2]
        return 1.0 - jnp.abs(jnp.vdot(target, psi_final)) ** 2

    idx_grid = jnp.arange(n_grid)
    I1, I2 = jnp.meshgrid(idx_grid, idx_grid, indexing="ij")
    infidelities = np.array(jax.vmap(eval_grid_point)(I1.ravel(), I2.ravel()))
    best_idx = int(np.argmin(infidelities))
    best_init = np.array([grid[I1.ravel()[best_idx]], grid[I2.ravel()[best_idx]]])

    result = minimize(
        lambda x: float(eval_infidelity(jnp.array(x))),
        x0=best_init,
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-9},
    )
    v_phase1 = float((result.x[0] + np.pi) % (2.0 * np.pi) - np.pi)
    v_phase2 = float((result.x[1] + np.pi) % (2.0 * np.pi) - np.pi)
    fidelity_grid = 1.0 - infidelities.reshape(n_grid, n_grid)

    return {
        "v_phase1": v_phase1,
        "v_phase2": v_phase2,
        "infidelity": float(result.fun),
        "fidelity": 1.0 - float(result.fun),
        "grid_rad": np.array(grid),
        "fidelity_grid": fidelity_grid,
    }

