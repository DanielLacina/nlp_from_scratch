use rand::Rng;

#[derive(Clone)]
enum Activation {
    Relu,
    Sigmoid
}
struct Neuron {
    weights: Vec<f32>,
    activation: Activation
}

impl Neuron {
    pub fn new(num_weights: i32, activation: Activation) -> Self {
        let mut rng = rand::rng(); 
        let mut weights = Vec::new();
        for _ in (0..num_weights) {
            weights.push(rng.gen_range(-0.5..0.5));
        }
        return Self {
            weights,
            activation
        };
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_neurons: i32, input_size: i32, activation: Activation) -> Self {
        let mut neurons = Vec::new();
        for _ in (0..num_neurons) {
            neurons.push(Neuron::new(input_size, activation.clone()));
        }
        return Self {
            neurons
        };
    } 
}

pub struct NeuralNetwork {
    input_size: i32,
    layers: Vec<Layer>
} 

impl NeuralNetwork {
    pub fn new(input_size: i32) -> Self {
        let mut layers = Vec::new(); 
        layers.push(Layer::new(input_size, input_size, Activation::Relu));
        layers.push(Layer::new(1, 0, Activation::Sigmoid));
        return Self {
            input_size,
            layers
        }
    }
} 