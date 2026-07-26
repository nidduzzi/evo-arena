use std::ops::Mul;

#[derive(PartialEq, Debug)]
pub struct Matrix {
    data: Box<[f32]>,
    pub cols: usize,
    pub rows: usize,
}

impl Matrix {
    pub fn new(data: &[f32], rows: usize, cols: usize) -> Matrix {
        if data.len() != cols * rows {
            panic!("Expected cols*rows to be equal to the length of data.")
        } else {
            Matrix {
                data: data.into(),
                cols,
                rows,
            }
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
                "Expected lefthand-side matrix columns to be equal to right hand side matrix rows."
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
            data: out_data.into_boxed_slice(),
            rows: out_rows,
            cols: out_cols,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

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
        let c = &a * &b;
    }

    #[test]
    #[should_panic]
    fn test_mismatch_mat_constructor() {
        let data_a: [f32; 4] = [1., 2., 3., 4.];
        let a = Matrix::new(&data_a, 2, 3);
    }
}
