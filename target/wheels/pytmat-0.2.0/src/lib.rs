use pyo3::prelude::*;
use numpy::ndarray::{Array1, Array2};
use numpy::{Complex64, PyArray1, PyArray2, PyReadonlyArrayDyn, PyReadonlyArray1, ToPyArray};

use tmatrix::Data;

#[pyclass]
struct DataPy {
    d: Array1<f64>,
    n: Array2<Complex64>,
    wl: Array1<f64>,
    theta: f64,
    phi: f64
}
#[pymethods]
impl DataPy {
    #[new]
    fn new(
        d: PyReadonlyArray1<f64>, 
        n: PyReadonlyArrayDyn<Complex64>,
        wl: PyReadonlyArray1<f64>,
        theta: f64,
        phi: f64) -> Self {
        let n_array = n.as_array();
        let n_2d = n_array.into_dimensionality::<numpy::ndarray::Dim<[usize;2]>>()
            .expect("Failed to convert n to 2D array");
        // Ensure that d is a 1D array
        DataPy { d: d.as_array().to_owned(), n: n_2d.to_owned(), wl: wl.as_array().to_owned(), theta: theta, phi: phi }
    }
    #[getter]
    fn get_d<'py>(&self,py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.d.to_pyarray(py))
    }
    #[getter]
    fn get_n<'py>(&self,py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        Ok(self.n.to_pyarray(py))
    }
    #[getter]
    fn get_wl<'py>(&self,py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.wl.to_pyarray(py))
    }
    #[getter]
    fn get_theta(&self) -> PyResult<f64> {
        Ok(self.theta)
    }
    #[getter]
    fn get_phi(&self) -> PyResult<f64> {
        Ok(self.phi)
    }

    fn simulate(&self) -> PyResult<Simulation> {
        // let len = arr.len();
        let output_array = Data::new(
            self.d.clone(), 
            self.n.clone(), 
            self.wl.clone(), 
            self.theta, self.phi);

        let r_out = output_array.get_r_power_vec();
        let t_out = output_array.get_t_power_vec();
        

        Ok(Simulation::new(
            Array1::from(t_out),
            Array1::from(r_out)))
    }
}

#[pyclass]
struct Simulation {
    t: Array1<f64>,
    r: Array1<f64>
}
#[pymethods]
impl Simulation {
    #[getter]
    fn get_t<'py>(&self,py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.t.to_pyarray(py))
    }
    #[getter]
    fn get_r<'py>(&self,py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.r.to_pyarray(py))
    }
}

impl Simulation {
    fn new(t: Array1<f64>, r: Array1<f64>) -> Self {
        Simulation { t, r }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn pytmat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DataPy>()?;
    m.add_class::<Simulation>()?;
    Ok(())
}