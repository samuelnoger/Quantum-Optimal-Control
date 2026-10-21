import jax.numpy as jnp
import jax
import numpy as np

def generate_analytic_pulse(duration_ns, dt_ns, anharmonicity_mhz, pulse_type='drag', target_angle=np.pi):
    """Generates textbook Gaussian or DRAG pulses for arbitrary rotations."""
    t = np.arange(0, duration_ns, dt_ns)
    mu = duration_ns / 2
    sigma = duration_ns / 4  # Standard 4-sigma width to fit the window
    
    # Base Gaussian (I-channel)
    I_pulse = np.exp(-0.5 * ((t - mu) / sigma)**2)
    I_pulse -= I_pulse[0]  # Force boundaries perfectly to zero
    
    # Normalize total area to the exact target angle (e.g., pi/2 for 90 degrees)
    area = np.sum(I_pulse) * dt_ns
    I_pulse = I_pulse * (target_angle / area)
    
    # Q-channel Correction
    if pulse_type == 'drag':
        anharmonicity_ghz = anharmonicity_mhz / 1000.0
        dI_dt = np.gradient(I_pulse, dt_ns)
        # Standard first-order DRAG correction
        Q_pulse = 0.5 * dI_dt / (2 * np.pi * anharmonicity_ghz)
    else:
        Q_pulse = np.zeros_like(I_pulse)
        
    return I_pulse, Q_pulse

def make_gaussian_kernel(dt, cutoff_freq_ghz):
    """Built ONCE with numpy, outside any jit: its length depends on dt and the cutoff, so it must be concrete."""
    sigma = 1.0 / (2 * np.pi * cutoff_freq_ghz)               # ns
    m = int(np.ceil(4 * sigma / dt))
    t = np.arange(-m, m + 1) * dt                              # symmetric grid around 0
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return jnp.asarray(k / k.sum())

def apply_low_pass_filter(raw_flux_pulse, kernel):
    m = (kernel.shape[0] - 1) // 2
    padded = jnp.pad(raw_flux_pulse, (m, m), mode="edge")      # hold the end values; mode='same' would pad with zeros
    return jnp.convolve(padded, kernel, mode="valid")