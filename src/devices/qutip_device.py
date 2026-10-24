import numpy as np
import qutip as qt

class HiddenDevice:
    def __init__(self, levels=3, true_coupling=21.5, true_detuning=595.0, true_anharm=-305.0, t1_ns=50000.0, t2_ns=30000.0):
        """
        A simulated physical fridge with 'secret' hardware parameters that differ from the JAX digital twin.
        """
        self.levels = levels
        self.dt = 0.1  # Hardware AWG resolution (ns)
        
        # --- Build QuTiP Operators ---
        b = qt.destroy(levels)
        bdag = b.dag()
        eye = qt.qeye(levels)
        num = bdag * b

        H1 = 0.5 * 2 * np.pi * (true_anharm / 1000.0) * num * (num - eye)
        H2 = 0.5 * 2 * np.pi * (true_anharm / 1000.0) * num * (num - eye)
        
        # Static exchange coupling
        H_int = 2 * np.pi * (true_coupling / 1000.0) * (qt.tensor(b, bdag) + qt.tensor(bdag, b))
        
        # Z-control for qubit 1
        self.Hz_control = qt.tensor(num, eye)
        
        # Full static Hamiltonian
        self.H0 = qt.tensor(H1, eye) + qt.tensor(eye, H2) + H_int + 2 * np.pi * (true_detuning / 1000.0) * self.Hz_control

        # --- Local Drive Operators ---
        Hx = (b + bdag) / 2
        Hy = (b - bdag) / (2j)
        self.Hx1, self.Hy1 = qt.tensor(Hx, eye), qt.tensor(Hy, eye)
        self.Hx2, self.Hy2 = qt.tensor(eye, Hx), qt.tensor(eye, Hy)
        
        # --- Collapse Operators for Decoherence ---
        self.c_ops = []
        if t1_ns > 0:
            gamma_1 = 1.0 / t1_ns
            self.c_ops.append(np.sqrt(gamma_1) * qt.tensor(b, eye))
            self.c_ops.append(np.sqrt(gamma_1) * qt.tensor(eye, b))
        if t2_ns > 0:
            gamma_2 = 1.0 / t2_ns
            self.c_ops.append(np.sqrt(gamma_2) * qt.tensor(num, eye))
            self.c_ops.append(np.sqrt(gamma_2) * qt.tensor(eye, num))

    def play_pulse(self, flux_array, initial_state):
        n = len(flux_array)
        # n steps of exactly dt, matching JAX
        tlist = np.arange(n + 1) * self.dt 

        # order=0 forces piecewise-constant interpolation
        coeff = qt.coefficient(np.append(flux_array, flux_array[-1]), tlist=tlist, order=0)
        H_total = qt.QobjEvo([self.H0, [self.Hz_control, coeff]])

        result = qt.mesolve(
            H_total, 
            initial_state, 
            tlist, 
            c_ops=self.c_ops, 
            options={"max_step": self.dt / 2}
        )
        return result.states[-1]

    def measure_populations(self, flux_array, initial_state, n_shots=2000, rng=None):
        """
        Plays the pulse and simulates projective readout shot noise over n_shots.
        Returns the measured populations in the computational basis: [|00>, |01>, |10>, |11>].
        """
        rho_final = self.play_pulse(flux_array, initial_state)
        
        # Extract exact diagonal probabilities across all 9 levels (3x3 system)
        exact_probs = np.clip(rho_final.diag().real, 0.0, 1.0)
        exact_probs /= np.sum(exact_probs)  # Ensure probabilities sum to 1.0
        
        # If n_shots is None, return exact noiseless probabilities
        if n_shots is None:
            return exact_probs[[0, 1, 3, 4]]
            
        # Sample multinomial projective measurements
        if rng is None:
            counts = np.random.multinomial(n_shots, exact_probs)
        else:
            counts = rng.multinomial(n_shots, exact_probs)
            
        noisy_probs = counts / float(n_shots)
        return noisy_probs[[0, 1, 3, 4]]

    def play_sequence(self, wf, initial_state):
        """wf: dict of equal-length arrays I1,Q1,I2,Q2,flux in rad/ns (zeros where idle)."""
        n = len(wf["flux"])
        tlist = np.arange(n + 1) * self.dt
    
        def c(x):
            x = np.asarray(x, dtype=float)
            return qt.coefficient(np.append(x, x[-1]), tlist=tlist, order=0)
    
        H = qt.QobjEvo([self.H0,
                        [self.Hz_control, c(wf["flux"])],
                        [self.Hx1, c(wf["I1"])], [self.Hy1, c(wf["Q1"])],
                        [self.Hx2, c(wf["I2"])], [self.Hy2, c(wf["Q2"])]])
        res = qt.mesolve(H, initial_state, tlist, c_ops=self.c_ops,
                         options={"max_step": self.dt / 2})
        return res.states[-1]