# src/optimization/training_loops.py
from functools import partial
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from src.physics.operators import build_two_qubit_operators
from src.control.pulses import make_gaussian_kernel
from src.optimization.losses import compute_cz_loss


def make_cz_optimizer(learning_rate: float, steps: int):
    """Creates an Adam optimizer with a cosine decay schedule and global norm clipping."""
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=steps, alpha=1e-3)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


def make_setup(
    config,
    n_harmonics: int = 8,
    key=None,
    coupling_mhz: float = None,
    detuning_mhz: float = None,
    anharmonicity_mhz: float = None,
):
    """
    Sets up the operators, Fourier sine basis, filter kernel, initial parameters, 
    and bound loss function for two-qubit CZ gate optimization.
    
    If calibrated parameters (coupling_mhz, detuning_mhz, anharmonicity_mhz) are not 
    provided, defaults to the nominal values in `config.hardware`.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    dt, T = config.pulse.dt_ns, config.pulse.duration_ns
    n_steps = int(round(T / dt))

    # Use calibrated parameters if passed, otherwise fall back to config defaults
    g_mhz = config.hardware.coupling_mhz if coupling_mhz is None else coupling_mhz
    d_mhz = config.hardware.idle_detuning_mhz if detuning_mhz is None else detuning_mhz
    a_mhz = config.hardware.anharmonicity_mhz if anharmonicity_mhz is None else anharmonicity_mhz

    H0, Hz = build_two_qubit_operators(
        levels=config.hardware.levels,
        anharm_1=a_mhz,
        anharm_2=a_mhz,
        coupling_mhz=g_mhz,
        idle_detuning_mhz=d_mhz,
    )

    x = (jnp.arange(n_steps) + 0.5) / n_steps
    basis = jnp.sin(jnp.pi * x[:, None] * jnp.arange(1, n_harmonics + 1)[None, :])
    kernel = make_gaussian_kernel(dt, config.pulse.cutoff_freq_ghz)
    
    f_max = 2 * jnp.pi * 0.6

    # Start near the |11> <-> |20> avoided crossing: lower qubit 1 by -(d_mhz + a_mhz)
    crossing_rad_ns = -2 * jnp.pi * ((d_mhz + a_mhz) / 1000.0)
    init = {
        'flux_coeffs': jnp.zeros(n_harmonics).at[0].set(crossing_rad_ns) + 0.05 * jax.random.normal(key, (n_harmonics,)),
        'phases': jnp.zeros(2),
    }

    loss_fn = partial(
        compute_cz_loss,
        dt=dt,
        H0=H0,
        Hz_control=Hz,
        basis=basis,
        kernel=kernel,
        f_max=f_max,
        penalty_weight=config.optimizer.leakage_weight,
    )
    return H0, Hz, loss_fn, init, basis, kernel, f_max


def train_pulse(loss_fn, init_params, optimizer, num_steps: int):
    """
    Executes the gradient descent loop to sculpt the pulse parameters.
    """
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def update_step(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    params = init_params
    history = {"loss": [], "infidelity": [], "leakage": []}

    for step in tqdm(range(num_steps), desc="Optimizing Pulse"):
        params, opt_state, loss, aux = update_step(params, opt_state)
        
        history["loss"].append(float(loss))
        history["infidelity"].append(float(aux["infidelity"]))
        history["leakage"].append(float(aux["leakage"]))
        
    return params, history