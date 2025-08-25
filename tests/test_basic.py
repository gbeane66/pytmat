import numpy as np
import pytmat

def test_reflection_transmission():
    d = np.array([200.0])
    N = 10
    n_air = np.full(N, 1.0, dtype=np.complex128)
    n_glass = np.full(N, 1.5, dtype=np.complex128)
    n = np.array([n_air, n_glass, n_air])
    wl = np.linspace(400, 700, N)
    theta = 0.0
    phi = 0.0
    data = pytmat.DataPy(d, n, wl, theta, phi)
    sim = data.simulate()
    assert np.all(sim.r >= 0)
    assert np.all(sim.t >= 0)
    assert np.allclose(sim.r + sim.t, 1, atol=0.1)  # Energy conservation