import numpy as np
import qutip as qt


def ry3(theta):
    """Ideal Ry(theta) rotation on the qubit subspace {|0>, |1>}, identity on |2>."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    U = np.eye(3, dtype=complex)
    U[:2, :2] = [[c, -s], [s, c]]
    return qt.Qobj(U)


def correlator(rho_framed, a, b, n_shots=5000, rng=None, postselect=True):
    """
    Measures <A(a) x B(b)> by rotating Q1 by Ry(-a) and Q2 by Ry(-b) 
    before projective Z-basis readout with finite shot noise.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    R = qt.tensor(ry3(-a), ry3(-b))
    rho_r = R * rho_framed * R.dag()
    
    # Extract probabilities in computational basis: [|00>, |01>, |10>, |11>]
    probs_9d = np.clip(rho_r.diag().real, 0.0, None)
    p_comp = probs_9d[[0, 1, 3, 4]]
    
    if postselect:
        p_comp = p_comp / np.sum(p_comp)
    else:
        # Without postselection, assign leaked shots to 0 parity contribution
        p_leak = max(0.0, 1.0 - np.sum(p_comp))
        p_all = np.append(p_comp, p_leak)
        p_all = p_all / np.sum(p_all)
        counts = rng.multinomial(n_shots, p_all)
        return float(counts[:4] @ np.array([+1, -1, -1, +1])) / n_shots

    counts = rng.multinomial(n_shots, p_comp)
    return float(counts @ np.array([+1, -1, -1, +1])) / n_shots


def run_chsh_experiment(rho_framed, n_shots=5000, seed=0, postselect=True):
    """
    Runs the 4-angle CHSH Bell inequality test on a frame-corrected 9x9 QuTiP state.
    Returns (S, S_std_err, E_dict).
    """
    rng = np.random.default_rng(seed)
    a, a2, b, b2 = 0.0, np.pi / 2.0, np.pi / 4.0, -np.pi / 4.0
    
    settings = {
        "E(a, b)": (a, b),
        "E(a, b')": (a, b2),
        "E(a', b)": (a2, b),
        "E(a', b')": (a2, b2),
    }
    
    E = {
        label: correlator(rho_framed, ang_a, ang_b, n_shots=n_shots, rng=rng, postselect=postselect)
        for label, (ang_a, ang_b) in settings.items()
    }
    
    S = E["E(a, b)"] + E["E(a, b')"] + E["E(a', b)"] - E["E(a', b')"]
    err = float(np.sqrt(sum((1.0 - e ** 2) / n_shots for e in E.values())))
    return float(S), err, E