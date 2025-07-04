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

struct TCoefficients {
    t_te: ComplexF64,
    t_tm: ComplexF64,
    t_total: ComplexF64
}
impl TCoefficients {
    #[inline(always)]
    fn new(t_te: ComplexF64, t_tm: ComplexF64, phi: f64) -> Self {
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let t_total = t_te * cos_phi.powi(2) + t_tm * sin_phi.powi(2);
        TCoefficients { t_te, t_tm, t_total }
    }
    #[inline(always)]
    fn get_t_te(&self) -> &ComplexF64 {
        &self.t_te
    }
    #[inline(always)]
    fn get_t_tm(&self) -> &ComplexF64 {
        &self.t_tm
    }
    #[inline(always)]
    fn get_t_total(&self) -> &ComplexF64 {
        &self.t_total
    }
}

struct RCoefficients {
    r_te: ComplexF64,
    r_tm: ComplexF64,
    r_total: ComplexF64
}
impl RCoefficients {
    #[inline(always)]
    pub fn new(r_te: ComplexF64, r_tm: ComplexF64, phi: f64) -> Self {
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let r_total = r_te * cos_phi.powi(2) + r_tm * sin_phi.powi(2);
        RCoefficients { r_te, r_tm, r_total }
    }
    #[inline(always)]
    fn get_r_te(&self) -> &ComplexF64 {
        &self.r_te
    }
    #[inline(always)]
    fn get_r_tm(&self) -> &ComplexF64 {
        &self.r_tm
    }
    #[inline(always)]
    fn get_r_total(&self) -> &ComplexF64 {
        &self.r_total
    }
}

struct Transfer {
    t_final_te: LayerMatrix,
    t_final_tm: LayerMatrix
}
impl Transfer {
    #[inline(always)]
    fn new(t_final_te: LayerMatrix, t_final_tm: LayerMatrix) -> Self {
        Transfer { t_final_te, t_final_tm }
    }

    #[inline(always)]
    fn get_r(&self, phi: f64) -> RCoefficients {
        RCoefficients::new( 
            self.t_final_te[(1, 0)]/self.t_final_te[(0, 0)], 
            self.t_final_tm[(1, 0)]/self.t_final_tm[(0, 0)],
            phi)
    }

    #[inline(always)]
    fn get_t(&self, phi: f64) -> TCoefficients {
        TCoefficients::new( 
            Complex::new(1.0, 0.0) / self.t_final_te[(0, 0)], 
            Complex::new(1.0, 0.0) / self.t_final_tm[(0, 0)],
            phi)
    }

    #[inline(always)]
    fn get_t_power(
        &self,
        n: &Array2<Complex<f64>>,
        theta: f64,
        phi: f64,
        j: usize) -> f64 {
        let n_initial = n[(0, j)];
        let n_final = n[(n.shape()[0] - 1, j)];

        let n_initial_costheta = n_initial * theta.cos();

        let n_final_costheta_n = (n_final * n_final - n_initial * n_initial * (theta.sin()).powi(2)).sqrt();

        let scaling_factor = n_final_costheta_n / n_initial_costheta;

        let cos2phi = phi.cos().powi(2);
        let sin2phi = phi.sin().powi(2);

        let t_te = Complex::new(1.0,0.0) / self.t_final_te[(0,0)];
        let t_tm = Complex::new(1.0,0.0) / self.t_final_tm[(0,0)];

        (scaling_factor.re) * (cos2phi * t_te.abs().powi(2) + sin2phi * t_tm.abs().powi(2))
    }
    #[inline(always)]
    fn get_r_power(&self, phi: f64) -> f64 {

        let cos2phi = phi.cos().powi(2);
        let sin2phi = phi.sin().powi(2);

        let r_te = self.t_final_te[(1, 0)] / self.t_final_te[(0, 0)];
        let r_tm = self.t_final_tm[(1, 0)] / self.t_final_tm[(0, 0)];

        cos2phi * r_te.abs().powi(2) + sin2phi * r_tm.abs().powi(2)
        // r_tm.abs().powi(2)
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
    pub fn new(d: Array1<f64>, n: Array2<ComplexF64>, wl: Array1<f64>, theta: f64, phi: f64) -> Self {
        Data { d, n, wl, theta, phi, transfer_cache: OnceCell::new() } // Default phi to 0 for TE polarization
    }

    pub fn get_d(&self) -> &Array1<f64> {
        &self.d
    }
    pub fn get_n(&self) -> &Array2<ComplexF64> {
        &self.n
    }
    pub fn get_wl(&self) -> &Array1<f64> {
        &self.wl
    }
    pub fn get_theta(&self)-> &f64 {
        &self.theta
    }
    pub fn get_phi(&self)-> &f64 {
        &self.phi
    }

    pub fn transfer_for_wavelength(&self, j: usize, pol: Polarization) -> LayerMatrix {

        let d = self.get_d();
        let n = self.get_n();
        let wl = self.get_wl();
        let theta = *self.get_theta();

        let wl_j = wl[j];
        let n0 = n[[0, j]];
        // let n1 = n[[1, j]];
        let nsin_theta0 = (n0 * theta.sin()).powi(2);

        let cos_theta = |ni: ComplexF64| (ni*ni - nsin_theta0).sqrt()/ni;

        let r_t = |ni: ComplexF64, nip1: ComplexF64, cos_ni: ComplexF64, cos_nip1: ComplexF64| match pol {
            Polarization::TE => (
                (ni * cos_ni - nip1 * cos_nip1) / (ni * cos_ni + nip1 * cos_nip1),
                (2.0 * ni * cos_ni) / (ni * cos_ni + nip1 * cos_nip1),
            ),
            Polarization::TM => (
                (nip1 * cos_ni - ni * cos_nip1) / (nip1 * cos_ni + ni * cos_nip1),
                (2.0 * ni * cos_ni) / (nip1 * cos_ni + ni * cos_nip1),
            ),
        };

        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let two_pi = 2.0 * PI;

        let n1 = n[[1, j]];
        let cos_theta_0 = cos_theta(n0);
        let cos_theta_1 = cos_theta(n1);
        let (mut r, mut t) = r_t(n0, n1, cos_theta_0, cos_theta_1);

        let mut transfer_total = match pol {
            Polarization::TE => Matrix2::new(
                one/t,
                r/t,
                r/t,
                one/t,
            ),
            Polarization::TM => Matrix2::new(
                one/t,
                r/t,
                r/t,
                one/t,
            ),
        };

        for i in 1..n.shape()[0]-1 {
            let ni = n[[i, j]];
            let nip1 = n[[i + 1, j]];
            let cos_ni = cos_theta(ni);
            let cos_nip1 = cos_theta(nip1);

            let kz = (two_pi * cos_ni * ni) / wl_j;
            let exponent_pos = (IMAG*kz * d[i - 1]).exp();
            let exponent_neg = (-IMAG*kz * d[i - 1]).exp();

            let t_prop = Matrix2::new(
                exponent_neg,
                zero,
                zero,
                exponent_pos,
            );

            let (r_new, t_new) = r_t(ni, nip1, cos_ni, cos_nip1);

            r = r_new;
            t = t_new;

            let t_int = match pol {
                Polarization::TE => Matrix2::new(
                    one/t,
                    r/t,
                    r/t,
                    one/t,
                ),
                Polarization::TM => Matrix2::new(
                    one/t,
                    r/t,
                    r/t,
                    one/t,
                ),
            };

            transfer_total = transfer_total * t_prop * t_int;
        }
        transfer_total
    }

    pub fn transfer_for_wavelength_helper(
        d: &Array1<f64>,
        n: &Array2<ComplexF64>,
        wl: &Array1<f64>,
        theta: f64,
        j: usize, 
        pol: Polarization) -> LayerMatrix {

        let wl_j = wl[j];
        let n0 = n[[0, j]];
        // let n1 = n[[1, j]];
        let nsin_theta0 = (n0 * theta.sin()).powi(2);

        let cos_theta = |ni: ComplexF64| (ni*ni - nsin_theta0).sqrt()/ni;

        let r_t = |ni: ComplexF64, nip1: ComplexF64, cos_ni: ComplexF64, cos_nip1: ComplexF64| match pol {
            Polarization::TE => (
                (ni * cos_ni - nip1 * cos_nip1) / (ni * cos_ni + nip1 * cos_nip1),
                (2.0 * ni * cos_ni) / (ni * cos_ni + nip1 * cos_nip1),
            ),
            Polarization::TM => (
                (nip1 * cos_ni - ni * cos_nip1) / (nip1 * cos_ni + ni * cos_nip1),
                (2.0 * ni * cos_ni) / (nip1 * cos_ni + ni * cos_nip1),
            ),
        };

        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let two_pi = 2.0 * PI;

        let n1 = n[[1, j]];
        let cos_theta_0 = cos_theta(n0);
        let cos_theta_1 = cos_theta(n1);
        let (mut r, mut t) = r_t(n0, n1, cos_theta_0, cos_theta_1);

        let mut transfer_total = match pol {
            Polarization::TE => Matrix2::new(
                one/t,
                r/t,
                r/t,
                one/t,
            ),
            Polarization::TM => Matrix2::new(
                one/t,
                r/t,
                r/t,
                one/t,
            ),
        };

        for i in 1..n.shape()[0]-1 {
            let ni = n[[i, j]];
            let nip1 = n[[i + 1, j]];
            let cos_ni = cos_theta(ni);
            let cos_nip1 = cos_theta(nip1);

            let kz = (two_pi * cos_ni * ni) / wl_j;
            let exponent_pos = (IMAG*kz * d[i - 1]).exp();
            let exponent_neg = (-IMAG*kz * d[i - 1]).exp();

            let t_prop = Matrix2::new(
                exponent_neg,
                zero,
                zero,
                exponent_pos,
            );

            let (r_new, t_new) = r_t(ni, nip1, cos_ni, cos_nip1);

            r = r_new;
            t = t_new;

            let t_int = match pol {
                Polarization::TE => Matrix2::new(
                    one/t,
                    r/t,
                    r/t,
                    one/t,
                ),
                Polarization::TM => Matrix2::new(
                    one/t,
                    r/t,
                    r/t,
                    one/t,
                ),
            };

            transfer_total = transfer_total * t_prop * t_int;
        }
        transfer_total
    }

    pub fn transfer_calc(&self) -> Vec<Transfer> {

        let wl = &self.wl;
        let wl_len = wl.len();
        let d = &self.d;
        let n = &self.n;
        let theta = self.theta;

        let (te_transfers, tm_transfers):(Vec<_>, Vec<_>) = if wl_len > 100 {
            (
                (0..wl_len).into_par_iter().map(|j| Data::transfer_for_wavelength_helper(d,n,wl,theta,j,Polarization::TE)).collect(),
                (0..wl_len).into_par_iter().map(|j| Data::transfer_for_wavelength_helper(d,n,wl,theta,j,Polarization::TM)).collect(),
            )
        } else {
            (
                (0..wl_len).map(|j| self.transfer_for_wavelength(j,Polarization::TE)).collect(),
                (0..wl_len).map(|j| self.transfer_for_wavelength(j,Polarization::TM)).collect(),
            )
        };
        te_transfers.into_iter().zip(tm_transfers).map(|(te,tm)| Transfer::new(te,tm)).collect()
    }

    fn transfer_calc_cached(&self) -> &Vec<Transfer> {
        self.transfer_cache.get_or_init(|| self.transfer_calc())
    }

    pub fn get_r_power_vec(&self) -> Vec<f64> {
        let transfer_matrix = self.transfer_calc_cached();
        transfer_matrix.iter().map(|tm| tm.get_r_power(self.phi)).collect()
    }

    pub fn get_t_power_vec(&self) -> Vec<f64> {
        let transfer_matrix = self.transfer_calc_cached();
        transfer_matrix.iter().enumerate()
            .map(|(j, tm)| tm.get_t_power(self.get_n(), self.theta, self.phi, j)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}