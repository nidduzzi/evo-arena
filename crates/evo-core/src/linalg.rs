use std::ops::{Add, Mul, Sub};

#[derive(PartialEq, Debug, Clone)]
pub struct Matrix {
    data: std::rc::Rc<[f32]>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(data: &[f32], rows: usize, cols: usize) -> Matrix {
        if data.len() != cols * rows {
            panic!("Expected cols*rows to be equal to the length of data.")
        } else {
            Matrix {
                data: data.into(),
                rows,
                cols,
            }
        }
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Matrix {
        Matrix {
            data: self.data.iter().map(|&x| f(x)).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn zip_with(&self, other: &Matrix, f: impl Fn(f32, f32) -> f32) -> Matrix {
        if (self.rows != other.rows)
            || (self.cols != other.cols)
            || self.data.len() != other.data.len()
        {
            panic!("Expected this matrix to have the same dimensions as other matrix.")
        }
        Matrix {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        &self * rhs
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.cols != rhs.rows {
            panic!(
                "Expected lefthand-side matrix columns to be equal to righthand-side matrix rows."
            )
        }
        let out_rows: usize = self.rows;
        let out_cols: usize = rhs.cols;
        let out_numel: usize = out_rows * out_cols;
        let mut out_data = vec![0.0_f32; out_numel];
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..self.cols {
                    out_data[i * out_cols + j] +=
                        self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
                }
            }
        }
        Matrix {
            data: out_data.into(),
            rows: out_rows,
            cols: out_cols,
        }
    }
}

impl Add for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add<Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {
        self + &rhs
    }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        &self + rhs
    }
}

impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: Self) -> Self::Output {
        self.zip_with(rhs, |a, b| a + b)
    }
}

impl Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub<Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        &self - rhs
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Self) -> Self::Output {
        self.zip_with(rhs, |a, b| a - b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_mismatch_mat_constructor() {
        let data_a: [f32; 4] = [1., 2., 3., 4.];
        let _ = Matrix::new(&data_a, 2, 3);
    }

    #[test]
    fn test_matmul() {
        let data_a: [f32; 6] = [1., 2., 3., 4., 5., 6.];
        let a = Matrix::new(&data_a, 2, 3);
        let data_b: [f32; 6] = [7., 8., 9., 10., 11., 12.];
        let b = Matrix::new(&data_b, 3, 2);
        let c = a * b;
        let reference = Matrix::new(
            &[
                1. * 7. + 2. * 9. + 3. * 11.,
                1. * 8. + 2. * 10. + 3. * 12.,
                4. * 7. + 5. * 9. + 6. * 11.,
                4. * 8. + 5. * 10. + 6. * 12.,
            ],
            2,
            2,
        );

        assert_eq!(c, reference);
    }

    #[test]
    fn test_ref_matmul() {
        let data_a: [f32; 6] = [1., 2., 3., 4., 5., 6.];
        let a = Matrix::new(&data_a, 2, 3);
        let data_b: [f32; 6] = [7., 8., 9., 10., 11., 12.];
        let b = Matrix::new(&data_b, 3, 2);

        let reference = Matrix::new(
            &[
                1. * 7. + 2. * 9. + 3. * 11.,
                1. * 8. + 2. * 10. + 3. * 12.,
                4. * 7. + 5. * 9. + 6. * 11.,
                4. * 8. + 5. * 10. + 6. * 12.,
            ],
            2,
            2,
        );

        let c = &a * &b;
        assert_eq!(c, reference);

        let c = a * &b;
        assert_eq!(c, reference);

        let a = Matrix::new(&data_a, 2, 3);
        let c = &a * b;
        assert_eq!(c, reference);
    }

    #[test]
    #[should_panic]
    fn test_mismatch_matmul() {
        let data_a: [f32; 4] = [1., 2., 3., 4.];
        let a = Matrix::new(&data_a, 2, 2);
        let data_b: [f32; 6] = [7., 8., 9., 10., 11., 12.];
        let b = Matrix::new(&data_b, 3, 2);
        let _ = &a * &b;
    }

    #[test]
    fn test_add() {
        let data_a: [f32; 6] = [1., 2., 3., 4., 5., 6.];
        let a = Matrix::new(&data_a, 3, 2);
        let data_b: [f32; 6] = [7., 8., 9., 10., 11., 12.];
        let b = Matrix::new(&data_b, 3, 2);
        let c = a + b;
        let reference = Matrix::new(
            &[1. + 7., 2. + 8., 3. + 9., 4. + 10., 5. + 11., 6. + 12.],
            3,
            2,
        );

        assert_eq!(c, reference);
    }

    #[test]
    fn test_ref_add() {
        let data_a: [f32; 6] = [1., 2., 3., 4., 5., 6.];
        let a = Matrix::new(&data_a, 3, 2);
        let data_b: [f32; 6] = [7., 8., 9., 10., 11., 12.];
        let b = Matrix::new(&data_b, 3, 2);
        let reference = Matrix::new(
            &[1. + 7., 2. + 8., 3. + 9., 4. + 10., 5. + 11., 6. + 12.],
            3,
            2,
        );

        let c = &a + &b;
        assert_eq!(c, reference);

        let c = a + &b;
        assert_eq!(c, reference);

        let a = Matrix::new(&data_a, 3, 2);
        let c = &a + &b;
        assert_eq!(c, reference);
    }

    #[test]
    #[should_panic]
    fn test_mismatch_add() {
        let data_a: [f32; 4] = [1., 2., 3., 4.];
        let a = Matrix::new(&data_a, 2, 2);
        let data_b: [f32; 6] = [7., 8., 9., 10., 11., 12.];
        let b = Matrix::new(&data_b, 3, 2);
        let _ = &a + &b;
    }
}
