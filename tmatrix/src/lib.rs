// #![warn(missing_docs)]
//! Fast and efficient Transfer Matrix Method implementation.
//! 
//! Provides an intuitive and performant implementation of Fresnel equations for stacks of materials of varying refractive index and thickness over a range of different wavelengths.
//! $\mathrm{\phi}$
//! - Easy to use
//! - Use parralellization for consideration of wavelength arrays where the number of entries is larger than 100.

use nalgebra::Matrix2;

use num_complex::{Complex, ComplexFloat};
use ndarray::{Array1,Array2};
use std;
use once_cell::unsync::OnceCell;

use rayon::prelude::*;

const PI: f64 = std::f64::consts::PI;
const IMAG: Complex<f64> = Complex::new(0.0, 1.0);

// Define ComplexF64 and LayerMatrix types for convenience
type ComplexF64 = Complex<f64>;
type LayerMatrix = Matrix2<ComplexF64>;

/// An enum used to account for the two different TE and TM polarizations
pub enum Polarization {
    /// Transverse Electric
    TE, 
    /// Transverse Magnetic
    TM, 
}

// These coefficient structs were unused and removed to clean up dead code.
// The calculations they performed are now done directly inline where needed.

pub struct Transfer {
    t_final_te: LayerMatrix,
    t_final_tm: LayerMatrix
}
impl Transfer {
    #[inline(always)]
    pub fn new(te_matrix: LayerMatrix, tm_matrix: LayerMatrix) -> Self {
        Transfer { 
            t_final_te: te_matrix, 
            t_final_tm: tm_matrix 
        }
    }

    /// Calculate transmission power for given conditions
    #[inline(always)]
    pub fn get_t_power(
        &self,
        refractive_indices: &Array2<Complex<f64>>,
        incident_angle: f64,
        polarization_angle: f64,
        wavelength_index: usize
    ) -> f64 {
        let n_initial = refractive_indices[(0, wavelength_index)];
        let n_final = refractive_indices[(refractive_indices.shape()[0] - 1, wavelength_index)];

        let n_initial_costheta = n_initial * incident_angle.cos();
        let sin_theta_squared = incident_angle.sin().powi(2);
        let n_final_costheta_n = (n_final * n_final - n_initial * n_initial * sin_theta_squared).sqrt();

        let scaling_factor = n_final_costheta_n / n_initial_costheta;

        let cos2phi = polarization_angle.cos().powi(2);
        let sin2phi = polarization_angle.sin().powi(2);

        let t_te = Complex::new(1.0, 0.0) / self.t_final_te[(0, 0)];
        let t_tm = Complex::new(1.0, 0.0) / self.t_final_tm[(0, 0)];

        (scaling_factor.re) * (cos2phi * t_te.abs().powi(2) + sin2phi * t_tm.abs().powi(2))
    }

    /// Calculate reflection power for given polarization
    #[inline(always)]
    pub fn get_r_power(&self, polarization_angle: f64) -> f64 {
        let cos2phi = polarization_angle.cos().powi(2);
        let sin2phi = polarization_angle.sin().powi(2);

        let r_te = self.t_final_te[(1, 0)] / self.t_final_te[(0, 0)];
        let r_tm = self.t_final_tm[(1, 0)] / self.t_final_tm[(0, 0)];

        cos2phi * r_te.abs().powi(2) + sin2phi * r_tm.abs().powi(2)
    }
}

pub struct Data {
    d: Array1<f64>, // Thickness of each layer
    n: Array2<Complex<f64>>, // Refractive indices of each layer at different wavelengths
    wl: Array1<f64>, // Wavelengths at which the refractive indices are defined
    theta: f64, // Angle of incidence in radians
    phi: f64, // Polarization angle in radians
    transfer_cache: OnceCell<Vec<Transfer>>
}

impl Data {
    /// Create a new multilayer optical data structure
    pub fn new(
        layer_thicknesses: Array1<f64>, 
        refractive_indices: Array2<ComplexF64>, 
        wavelengths: Array1<f64>, 
        incident_angle_rad: f64, 
        polarization_angle_rad: f64
    ) -> Self {
        Data { 
            d: layer_thicknesses, 
            n: refractive_indices, 
            wl: wavelengths, 
            theta: incident_angle_rad, 
            phi: polarization_angle_rad, 
            transfer_cache: OnceCell::new() 
        }
    }

    /// Get layer thicknesses
    pub fn layer_thicknesses(&self) -> &Array1<f64> {
        &self.d
    }
    
    /// Get refractive indices array (layers × wavelengths)
    pub fn refractive_indices(&self) -> &Array2<ComplexF64> {
        &self.n
    }
    
    /// Get wavelength array
    pub fn wavelengths(&self) -> &Array1<f64> {
        &self.wl
    }
    
    /// Get incident angle in radians
    pub fn incident_angle(&self) -> f64 {
        self.theta
    }
    
    /// Get polarization angle in radians (0=TE, π/2=TM)
    pub fn polarization_angle(&self) -> f64 {
        self.phi
    }

    // Legacy getter methods for backward compatibility
    pub fn get_d(&self) -> &Array1<f64> {
        &self.d
    }
    pub fn get_n(&self) -> &Array2<ComplexF64> {
        &self.n
    }
    pub fn get_wl(&self) -> &Array1<f64> {
        &self.wl
    }
    pub fn get_theta(&self) -> &f64 {
        &self.theta
    }
    pub fn get_phi(&self) -> &f64 {
        &self.phi
    }

    /// Calculate transfer matrix for a specific wavelength index and polarization
    pub fn calculate_transfer_matrix_for_wavelength(
        d: &Array1<f64>,
        n: &Array2<ComplexF64>,
        wl: &Array1<f64>,
        theta: f64,
        j: usize, 
        polarization: Polarization
    ) -> LayerMatrix {
        let wavelength = wl[j];
        let n0 = n[[0, j]];
        let nsin_theta0_squared = (n0 * theta.sin()).powi(2);

        // Calculate cosine of angle for each layer
        let cos_angle = |ni: ComplexF64| (ni * ni - nsin_theta0_squared).sqrt() / ni;

        // Calculate reflection and transmission coefficients based on polarization
        let calc_coefficients = |ni: ComplexF64, nip1: ComplexF64, cos_ni: ComplexF64, cos_nip1: ComplexF64| {
            match polarization {
                Polarization::TE => (
                    (ni * cos_ni - nip1 * cos_nip1) / (ni * cos_ni + nip1 * cos_nip1),
                    (2.0 * ni * cos_ni) / (ni * cos_ni + nip1 * cos_nip1),
                ),
                Polarization::TM => (
                    (nip1 * cos_ni - ni * cos_nip1) / (nip1 * cos_ni + ni * cos_nip1),
                    (2.0 * ni * cos_ni) / (nip1 * cos_ni + ni * cos_nip1),
                ),
            }
        };

        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let two_pi = 2.0 * PI;

        // Initialize with first interface
        let n1 = n[[1, j]];
        let cos_theta_0 = cos_angle(n0);
        let cos_theta_1 = cos_angle(n1);
        let (mut r, mut t) = calc_coefficients(n0, n1, cos_theta_0, cos_theta_1);

        let mut transfer_total = Matrix2::new(one / t, r / t, r / t, one / t);

        // Process intermediate layers
        for i in 1..n.shape()[0] - 1 {
            let ni = n[[i, j]];
            let nip1 = n[[i + 1, j]];
            let cos_ni = cos_angle(ni);
            let cos_nip1 = cos_angle(nip1);

            // Propagation matrix through the layer
            let kz = (two_pi * cos_ni * ni) / wavelength;
            let layer_thickness = d[i - 1];
            let exponent_pos = (IMAG * kz * layer_thickness).exp();
            let exponent_neg = (-IMAG * kz * layer_thickness).exp();

            let propagation_matrix = Matrix2::new(exponent_neg, zero, zero, exponent_pos);

            // Interface matrix
            let (r_new, t_new) = calc_coefficients(ni, nip1, cos_ni, cos_nip1);
            r = r_new;
            t = t_new;

            let interface_matrix = Matrix2::new(one / t, r / t, r / t, one / t);

            transfer_total = transfer_total * propagation_matrix * interface_matrix;
        }
        
        transfer_total
    }

    pub fn transfer_for_wavelength(&self, j: usize, polarization: Polarization) -> LayerMatrix {
        Self::calculate_transfer_matrix_for_wavelength(
            &self.d, &self.n, &self.wl, self.theta, j, polarization
        )
    }

    /// Calculate all transfer matrices for all wavelengths
    /// Uses parallelization for large wavelength arrays (>100 elements)
    pub fn transfer_calc(&self) -> Vec<Transfer> {
        let wl_len = self.wl.len();
        let use_parallel = wl_len > 100;

        let (te_transfers, tm_transfers): (Vec<_>, Vec<_>) = if use_parallel {
            // Clone necessary data for parallel processing to avoid borrowing issues
            let d = self.d.clone();
            let n = self.n.clone();
            let wl = self.wl.clone();
            let theta = self.theta;
            
            (0..wl_len)
                .into_par_iter()
                .map(|j| {
                    let te = Self::calculate_transfer_matrix_for_wavelength(
                        &d, &n, &wl, theta, j, Polarization::TE
                    );
                    let tm = Self::calculate_transfer_matrix_for_wavelength(
                        &d, &n, &wl, theta, j, Polarization::TM
                    );
                    (te, tm)
                })
                .unzip()
        } else {
            // Use sequential processing for smaller arrays
            (0..wl_len)
                .map(|j| {
                    let te = self.transfer_for_wavelength(j, Polarization::TE);
                    let tm = self.transfer_for_wavelength(j, Polarization::TM);
                    (te, tm)
                })
                .unzip()
        };

        te_transfers
            .into_iter()
            .zip(tm_transfers)
            .map(|(te, tm)| Transfer::new(te, tm))
            .collect()
    }

    /// Get cached transfer matrices, computing if necessary
    fn get_transfer_matrices(&self) -> &Vec<Transfer> {
        self.transfer_cache.get_or_init(|| self.transfer_calc())
    }

    /// Calculate reflection power spectrum across all wavelengths
    pub fn reflection_spectrum(&self) -> Vec<f64> {
        let transfer_matrices = self.get_transfer_matrices();
        transfer_matrices
            .iter()
            .map(|tm| tm.get_r_power(self.phi))
            .collect()
    }

    /// Calculate transmission power spectrum across all wavelengths
    pub fn transmission_spectrum(&self) -> Vec<f64> {
        let transfer_matrices = self.get_transfer_matrices();
        transfer_matrices
            .iter()
            .enumerate()
            .map(|(j, tm)| tm.get_t_power(&self.n, self.theta, self.phi, j))
            .collect()
    }

    // Legacy methods for backward compatibility
    pub fn get_r_power_vec(&self) -> Vec<f64> {
        self.reflection_spectrum()
    }

    pub fn get_t_power_vec(&self) -> Vec<f64> {
        self.transmission_spectrum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_creation() {
        let d = Array1::from(vec![200.0]);
        let n = Array2::from_shape_vec(
            (3, 1), 
            vec![Complex::new(1.0, 0.0), Complex::new(1.5, 0.0), Complex::new(1.0, 0.0)]
        ).unwrap();
        let wl = Array1::from(vec![500.0]);
        let theta = 0.0;
        let phi = 0.0;
        
        let data = Data::new(d, n, wl, theta, phi);
        assert_eq!(data.get_d().len(), 1);
        assert_eq!(data.get_n().shape(), &[3, 1]);
        assert_eq!(*data.get_theta(), 0.0);
        assert_eq!(*data.get_phi(), 0.0);
    }
}