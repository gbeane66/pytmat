import numpy as np
import pytmat
import matplotlib.pyplot as plt

# Define layer thicknesses (nm)
d = np.array([200.0])  # Example: 3 layers, with two seminfinite layers on either side of a 200 nm layer.

# Define complex refractive indices for each layer at each wavelength
# Shape: (num_layers, num_wavelengths)
# Define number of wavelengths (300 for example)
N = 300
# Define refractive indices for 3 layers (air, glass, air) at 300 wavelengths
# Here, we assume air (n=1.0) and a glass layer (n=1.5)
# The first and last layers are seminfinite air layers, so their refractive index is 1.0
# The middle layer is a glass layer with a refractive index of 1.5
n_air = np.full(N, 1.0, dtype=np.complex128)
n_glass = np.full(N, 1.5, dtype=np.complex128)
n = np.array([n_air,n_glass,n_air])

# Wavelengths (nm)
wl = np.linspace(400, 700, N)

# Angle of incidence (radians) and polarization angle (radians)
theta = 0.0  # normal incidence
phi = 0.0    # TE polarization

# Create the TMM data object
data = pytmat.DataPy(d, n, wl, theta, phi)

# Simulate the multilayer stack
simulation = data.simulate()

# Compute reflection and transmission spectra
R, T = simulation.r, simulation.t

fig, ax = plt.subplots()
ax.plot(wl, R, label='Reflection',color='blue')
ax.plot(wl, T, label='Transmission', color='orange')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance / Transmittance')
plt.title('Multilayer Stack Reflection and Transmission')
ax.legend()
plt.show()