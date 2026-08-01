use crate::linalg::Matrix;

#[derive(Clone, Debug)]
pub enum Activation {
    Sigmoid,
}

struct Layer {
    weight: Matrix,
    bias: Matrix,
    activation: Activation,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Layer {
    pub(crate) fn activate(&self, input: &Matrix) -> Matrix {
        let mut z = (&self.weight * input);
        z = z + &self.bias;
        match self.activation {
            Activation::Sigmoid => z.map(sigmoid),
        }
    }

    pub(crate) fn new<R: rand::Rng + ?Sized>(
        input: usize,
        output: usize,
        activation: Activation,
        rng: &mut R,
    ) -> Layer {
        Layer {
            weight: Matrix::from_dist(
                rand_distr::Normal::new(0.0, 0.1).unwrap(),
                rng,
                output,
                input,
            ),
            bias: Matrix::from_dist(rand_distr::Normal::new(0.0, 0.05).unwrap(), rng, output, 1),
            activation,
        }
    }
}

pub struct NeuralNet {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub enum NeuralNetError {
    LayerDimMissmatch {
        layer: usize,
        expected: usize,
        got: usize,
    },
    InputSizeMissmatch {
        expected: usize,
        got: usize,
    },
    EmptyNetwork,
}

impl std::fmt::Display for NeuralNetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            NeuralNetError::LayerDimMissmatch {
                layer,
                expected,
                got,
            } => write!(
                f,
                "Expected layer {} to get {} but got {}.",
                layer, expected, got,
            ),
            NeuralNetError::EmptyNetwork => write!(
                f,
                "Expected a neural network with at least one layer. But got an empty network."
            ),
            NeuralNetError::InputSizeMissmatch { expected, got } => {
                write!(f, "Expected input of size {} but got {}", expected, got)
            }
        }
    }
}
impl std::error::Error for NeuralNetError {}

impl NeuralNet {
    pub fn forward(&self, input: &Matrix) -> Result<Matrix, NeuralNetError> {
        if self.layers.len() < 1 {
            return Err(NeuralNetError::EmptyNetwork);
        }
        if self.layers[0].weight.cols != input.rows {
            return Err(NeuralNetError::InputSizeMissmatch {
                expected: self.layers[0].weight.cols,
                got: input.rows,
            });
        }

        let mut current = input.clone();
        for layer in self.layers.iter() {
            current = layer.activate(&current);
        }

        Ok(current)
    }

    pub fn from_dense_sizes_with_rng<R: rand::Rng + ?Sized>(
        layers: &[usize],
        activation: Activation,
        rng: &mut R,
    ) -> NeuralNet {
        let layer_sizes = layers.iter().zip(layers.iter().skip(1));
        NeuralNet {
            layers: layer_sizes
                .clone()
                .map(|(input, output)| Layer::new(*input, *output, activation.clone(), rng))
                .collect(),
        }
    }

    pub fn from_dense_sizes(layers: &[usize], activation: Activation) -> NeuralNet {
        Self::from_dense_sizes_with_rng(layers, activation, &mut rand::rng())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_init() {
        let layer_sizes = [3, 4, 2];
        let nn = NeuralNet::from_dense_sizes(&layer_sizes, Activation::Sigmoid);
        assert_eq!(nn.layers.len(), 2);

        // Check input shapes
        assert_eq!(
            nn.layers
                .iter()
                .map(|layer| layer.weight.cols)
                .collect::<Vec<usize>>(),
            layer_sizes
                .iter()
                .take(layer_sizes.len() - 1)
                .map(|v| *v)
                .collect::<Vec<usize>>()
        );

        // Check output shapes
        assert_eq!(
            nn.layers
                .iter()
                .map(|layer| layer.weight.rows)
                .collect::<Vec<usize>>(),
            layer_sizes
                .iter()
                .skip(1)
                .map(|v| *v)
                .collect::<Vec<usize>>()
        );
    }

    #[test]
    fn test_nn_forward() {
        let layer_sizes = [3, 4, 2];
        let nn = NeuralNet::from_dense_sizes(&layer_sizes, Activation::Sigmoid);
        let input = Matrix::from_dist(
            rand_distr::Normal::new(0.0, 1.0).unwrap(),
            &mut rand::rng(),
            3,
            1,
        );
        let output = nn.forward(&input);
        assert_eq!(output.is_ok(), true);
        let output = output.unwrap();
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 1);
    }
}
