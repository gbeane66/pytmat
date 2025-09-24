use pyo3::prelude::*;
use numpy::ndarray::{Array1, Array2};
use numpy::{Complex64, PyArray1, PyArray2, PyReadonlyArrayDyn, PyReadonlyArray1, ToPyArray};

use tmatrix::Data;

/// Python wrapper for optical multilayer data
#[pyclass]
struct DataPy {
    /// Layer thicknesses
    d: Array1<f64>,
    /// Complex refractive indices (layers × wavelengths)  
    n: Array2<Complex64>,
    /// Wavelength array
    wl: Array1<f64>,
    /// Incident angle in radians
    theta: f64,
    /// Polarization angle in radians (0=TE, π/2=TM)
    phi: f64
}
#[pymethods]
impl DataPy {
    /// Create a new multilayer optical structure
    /// 
    /// Parameters:
    /// - d: Layer thicknesses (1D array) - should have length = num_layers - 2
    /// - n: Complex refractive indices (2D array: layers × wavelengths)
    /// - wl: Wavelength array (1D array)
    /// - theta: Incident angle in radians
    /// - phi: Polarization angle in radians (0=TE, π/2=TM)
    #[new]
    fn new(
        d: PyReadonlyArray1<f64>, 
        n: PyReadonlyArrayDyn<Complex64>,
        wl: PyReadonlyArray1<f64>,
        theta: f64,
        phi: f64
    ) -> PyResult<Self> {
        let n_array = n.as_array();
        let n_2d = n_array.into_dimensionality::<numpy::ndarray::Dim<[usize;2]>>()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Refractive index array must be 2-dimensional (layers × wavelengths)"
            ))?;
        
        let d_array = d.as_array();
        let wl_array = wl.as_array();
        
        // Validate input dimensions
        let num_layers = n_2d.shape()[0];
        let num_wavelengths = n_2d.shape()[1];
        
        if d_array.len() != num_layers - 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Layer thickness array length ({}) must be num_layers - 2 ({}). \
                     First and last layers are semi-infinite.",
                    d_array.len(), num_layers - 2
                )
            ));
        }
        
        if wl_array.len() != num_wavelengths {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Wavelength array length ({}) must match refractive index wavelength dimension ({})",
                    wl_array.len(), num_wavelengths
                )
            ));
        }
        
        // Validate physical constraints
        if theta < 0.0 || theta > std::f64::consts::PI / 2.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Incident angle must be between 0 and π/2 radians"
            ));
        }
        
        if d_array.iter().any(|&thickness| thickness <= 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All layer thicknesses must be positive"
            ));
        }
        
        if wl_array.iter().any(|&wavelength| wavelength <= 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All wavelengths must be positive"
            ));
        }
        
        Ok(DataPy { 
            d: d_array.to_owned(), 
            n: n_2d.to_owned(), 
            wl: wl_array.to_owned(), 
            theta, 
            phi 
        })
    }
    /// Get layer thicknesses array
    #[getter]
    fn get_d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.d.to_pyarray(py))
    }
    
    /// Get layer thicknesses (alias for d)
    #[getter(layer_thicknesses)]
    fn get_layer_thicknesses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.d.to_pyarray(py))
    }
    
    /// Get complex refractive indices array
    #[getter]
    fn get_n<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        Ok(self.n.to_pyarray(py))
    }
    
    /// Get refractive indices (alias for n)
    #[getter(refractive_indices)]
    fn get_refractive_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        Ok(self.n.to_pyarray(py))
    }
    
    /// Get wavelength array
    #[getter]
    fn get_wl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.wl.to_pyarray(py))
    }
    
    /// Get wavelengths (alias for wl)
    #[getter(wavelengths)]
    fn get_wavelengths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.wl.to_pyarray(py))
    }
    
    /// Get incident angle in radians
    #[getter]
    fn get_theta(&self) -> PyResult<f64> {
        Ok(self.theta)
    }
    
    /// Get incident angle (alias for theta)
    #[getter(incident_angle)]
    fn get_incident_angle(&self) -> PyResult<f64> {
        Ok(self.theta)
    }
    
    /// Get polarization angle in radians  
    #[getter]
    fn get_phi(&self) -> PyResult<f64> {
        Ok(self.phi)
    }
    
    /// Get polarization angle (alias for phi)
    #[getter(polarization_angle)]
    fn get_polarization_angle(&self) -> PyResult<f64> {
        Ok(self.phi)
    }

    /// Run the transfer matrix simulation
    /// 
    /// Returns a Simulation object containing reflection and transmission spectra
    fn simulate(&self) -> PyResult<Simulation> {
        // Create the core Data structure with legacy constructor for now
        let optical_data = Data::new(
            self.d.clone(), 
            self.n.clone(), 
            self.wl.clone(), 
            self.theta, 
            self.phi
        );

        // Calculate reflection and transmission spectra
        let reflection_spectrum = optical_data.reflection_spectrum();
        let transmission_spectrum = optical_data.transmission_spectrum();

        Ok(Simulation::new(
            Array1::from(transmission_spectrum),
            Array1::from(reflection_spectrum)
        ))
    }
}

/// Simulation results containing optical spectra
#[pyclass]
struct Simulation {
    /// Transmission spectrum
    t: Array1<f64>,
    /// Reflection spectrum
    r: Array1<f64>
}
#[pymethods]
impl Simulation {
    /// Get transmission spectrum
    #[getter]
    fn get_t<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.t.to_pyarray(py))
    }
    
    /// Get transmission spectrum (alias for t)
    #[getter(transmission)]
    fn get_transmission<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.t.to_pyarray(py))
    }
    
    /// Get reflection spectrum
    #[getter]
    fn get_r<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.r.to_pyarray(py))
    }
    
    /// Get reflection spectrum (alias for r)
    #[getter(reflection)]
    fn get_reflection<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.r.to_pyarray(py))
    }
    
    /// Get absorbance spectrum (1 - transmission - reflection)
    #[getter(absorbance)]
    fn get_absorbance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let absorbance: Array1<f64> = Array1::from_iter(
            self.t.iter().zip(self.r.iter())
                .map(|(t, r)| 1.0 - t - r)
        );
        Ok(absorbance.to_pyarray(py))
    }
    
    /// Check energy conservation (R + T + A = 1)
    /// Returns the maximum deviation from unity
    fn energy_conservation_error(&self) -> f64 {
        self.t.iter().zip(self.r.iter())
            .map(|(t, r)| (1.0 - t - r).abs())
            .fold(0.0f64, |acc, x| acc.max(x))
    }
    
    /// Validate that energy is conserved within tolerance
    fn validate_energy_conservation(&self, tolerance: Option<f64>) -> PyResult<bool> {
        let tol = tolerance.unwrap_or(1e-6);
        let max_error = self.energy_conservation_error();
        Ok(max_error < tol)
    }
}

impl Simulation {
    /// Create a new simulation result
    fn new(transmission: Array1<f64>, reflection: Array1<f64>) -> Self {
        Simulation { t: transmission, r: reflection }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn pytmat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DataPy>()?;
    m.add_class::<Simulation>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;
    use numpy::Complex64;

    #[test]
    fn test_data_py_creation() {
        let d = array![200.0];
        let n = array![[Complex64::new(1.0, 0.0)], [Complex64::new(1.5, 0.0)], [Complex64::new(1.0, 0.0)]];
        let wl = array![500.0];
        let theta = 0.0;
        let phi = 0.0;
        let data = DataPy { d, n, wl, theta, phi };
        assert_eq!(data.d.len(), 1);
        assert_eq!(data.n.shape(), &[3, 1]);
    }
}