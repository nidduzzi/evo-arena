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
        let z = (&self.weight * input) + &self.bias;
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
            bias: Matrix::from_dist(rand_distr::Normal::new(0.0, 0.05).unwrap(), rng, 1, output),
            activation,
        }
    }
}

pub struct NeuralNet {
    layers: Vec<Layer>,
    scratch: Vec<Matrix>,
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
    pub fn forward(&mut self, input: &Matrix) -> Result<Matrix, NeuralNetError> {
        self.scratch[0] = input.clone();
        if self.layers[0].weight.cols != input.rows {
            return Err(NeuralNetError::InputSizeMissmatch {
                expected: self.layers[0].weight.cols,
                got: input.rows,
            });
        }

        for (i, layer) in self.layers.iter().enumerate() {
            self.scratch[i + 1] = layer.activate(&self.scratch[i]);
        }

        self.scratch
            .last()
            .ok_or(NeuralNetError::EmptyNetwork)
            .cloned()
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
            scratch: layer_sizes
                .map(|(input, output)| Matrix::from_const(0.0, *output, *input))
                .collect(),
        }
    }

    pub fn from_dense_sizes(layers: &[usize], activation: Activation) -> NeuralNet {
        Self::from_dense_sizes_with_rng(layers, activation, &mut rand::rng())
    }
}
