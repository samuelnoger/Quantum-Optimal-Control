"""Small consistency check for the digital-twin and device interfaces."""

import numpy as np
import qutip


def test_twin_and_device_expose_matching_flux_execution():
    from src.devices.qutip_device import HiddenDevice
    from src.physics.operators import build_two_qubit_operators
    from src.physics.propagators import propagate_flux

    device = HiddenDevice(t1_ns=0, t2_ns=0)
    H0, hz = build_two_qubit_operators(
        levels=3,
        anharm_1=-305.0,
        anharm_2=-305.0,
        coupling_mhz=21.5,
        idle_detuning_mhz=595.0,
    )
    pulse = np.zeros(4)
    twin_unitary = np.asarray(propagate_flux(pulse, device.dt, H0, hz))

    initial = qutip.ket2dm(qutip.tensor(qutip.basis(3, 1), qutip.basis(3, 0)))
    observed = device.play_pulse(pulse, initial)

    expected = twin_unitary @ np.asarray(initial.full()) @ twin_unitary.conj().T
    np.testing.assert_allclose(observed.full(), expected, atol=2e-5)
