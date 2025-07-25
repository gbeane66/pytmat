# 🌈 pyTMat

**pyTMat** is a blazing-fast, user-friendly Python package for simulating optical multilayer stacks using the Transfer Matrix Method (TMM). Powered by a Rust core for maximum performance, pyTMat makes it easy to compute reflection and transmission spectra for complex photonic structures — perfect for research, engineering, and education.

---

## 🚀 Features

- **Ultra-fast**: Rust-powered core for high-performance calculations
- **Easy-to-use**: Simple Python API, no need to write Rust
- **Parallelized**: Automatically uses all your CPU cores for large wavelength arrays
- **Polarization Support**: TE, TM, and arbitrary polarization angles
- **Complex Refractive Indices**: Supports lossy materials with complex indices
- **Flexible Layer Configurations**: Any number of layers with arbitrary thicknesses
- **Incident Angles**: Specify any angle of incidence
- **Wavelength Range**: Define custom wavelength ranges for your simulations
- **Numpy Integration**: Works seamlessly with NumPy arrays
- **Perfect for DBRs, filters, sensors, and more!**

---

## 📦 Installation

Install the latest release directly from [PyPI](https://pypi.org/project/pytmat/):

```bash
pip install pytmat
```

---

## 🧑‍💻 Quick Start

```python
import numpy as np
import pytmat

# Define layer thicknesses (nm)
d = np.array([100, 200, 100])  # Example: 3 layers

# Define complex refractive indices for each layer at each wavelength
# Shape: (num_layers, num_wavelengths)
n = np.array([
    [1.0+0j, 1.0+0j, 1.0+0j],   # Layer 1
    [2.0+0j, 2.0+0j, 2.0+0j],   # Layer 2
    [1.5+0j, 1.5+0j, 1.5+0j],   # Layer 3
])

# Wavelengths (nm)
wl = np.linspace(400, 700, 3)

# Angle of incidence (radians) and polarization angle (radians)
theta = 0.0  # normal incidence
phi = 0.0    # TE polarization

# Create the TMM data object
data = pytmat.Data(d, n, wl, theta, phi)

# Compute reflection and transmission spectra
R = data.get_r_power_vec()
T = data.get_t_power_vec()

print("Reflection:", R)
print("Transmission:", T)
```

---

## 📚 API Overview

- `Data(d, n, wl, theta, phi)`: Main class for defining your multilayer stack.
    - `d`: 1D array of layer thicknesses (float, nm or μm)
    - `n`: 2D array of complex refractive indices (layers × wavelengths)
    - `wl`: 1D array of wavelengths
    - `theta`: Angle of incidence (radians)
    - `phi`: Polarization angle (radians, 0=TE, π/2=TM)
- `get_r_power_vec()`: Returns reflection spectrum (array)
- `get_t_power_vec()`: Returns transmission spectrum (array)

---

## 🛠️ Advanced Usage

- **Arbitrary polarization**: Set `phi` between 0 (TE) and π/2 (TM) for mixed polarization.
- **Large arrays**: For >100 wavelengths, pyTMat automatically parallelizes computations.
- **Complex indices**: Supports lossy materials (complex n).

---

## 🏗️ Project Structure

```
pytmat/
├── tmatrix/         # Rust core library
├── pytmat/          # Python bindings (PyO3/maturin)
├── tests/           # Python tests
├── README.md
└── ...
```

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🌟 Acknowledgements

- Built with [Rust](https://www.rust-lang.org/) and [PyO3](https://pyo3.rs/)
- Uses [nalgebra](https://nalgebra.org/), [ndarray](https://docs.rs/ndarray/), and [rayon](https://docs.rs/rayon/)

---

## 🔗 Links

- [PyPI](https://pypi.org/project/pytmat/)
- [Documentation](https://your-docs-link-here)
- [Issues](https://github.com/yourusername/pytmat/issues)

---

> **pyTMat** — Fast, flexible, and fun multilayer optics for Python