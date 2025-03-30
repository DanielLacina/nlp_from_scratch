use rand::Rng;
use std::iter::zip;
use std::sync::Arc;
use crate::{embedding::{Embedding}, tokenizer::Tokenizer, utils::*};

#[derive(Clone)]
pub enum Activation {
    Relu,
    Sigmoid
}
struct Neuron {
    weights: Vec<f32>,
    bias: f32, 
    activation: Activation,
    last_input: Vec<f32>,
    activated_result: f32,
    unactivated_result: f32
}

impl Neuron {
    pub fn new(num_weights: i32, activation: Activation) -> Self {
        let mut rng = rand::rng(); 
        let mut weights = Vec::new();
        for _ in (0..num_weights) {
            weights.push(rng.gen_range(-0.5..0.5));
        }
        return Self {
            last_input: vec![],
            unactivated_result: 0.0,
            activated_result: 0.0,
            weights,
            activation,
            bias: 0.0
        };
    }

    pub fn calculate(&mut self, vector: &Vec<f32>) -> f32 {
        self.last_input = vector.clone();
        let mut sum = 0.0;
        for (value , weight ) in zip(vector, &self.weights) {
            sum += value * weight;
        } 
        self.unactivated_result = sum;
        let result = self.apply_activation(sum); 
        self.activated_result = result;
        return result;
    }

    pub fn get_weights_count(&self) -> i32 {
        return self.weights.len() as i32;
    }

    fn apply_activation(&self, value: f32) -> f32 {
        match self.activation {
            Activation::Relu => {
                return relu(value);
            }
            Activation::Sigmoid => {
                return sigmoid(value);
            }
            _ => panic!("invalid activation function")
        }
    }

    pub fn subtract_diff_with_respect_to_weights_from_weights(&mut self, initial_diff: f32, learning_rate: f32) {
        self.weights = zip(&self.last_input, &self.weights).map(|(input, weight)| weight - (*input * initial_diff * learning_rate)).collect();
    }

    pub fn subtract_diff_with_respect_to_bias_from_bias(&mut self, inital_diff: f32, learning_rate: f32) {
        self.bias -= inital_diff * learning_rate; 
    }

    pub fn calculate_diff_with_respect_to_input_neurons(&self, initial_diff: f32) -> Vec<f32> {
        return self.weights.clone().iter().map(|weight| weight * initial_diff).collect(); 
    }

    pub fn calculate_diff_of_activation_function(&self) -> f32 {
        let activation_diff = match &self.activation {
            Activation::Relu => {
                relu_diff(self.unactivated_result)
            }
            Activation::Sigmoid => {
                sigmoid_diff(self.unactivated_result)
            }
        };
        activation_diff
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_neurons: i32, num_weights: i32, activation: Activation) -> Self {
        let mut neurons = Vec::new();
        for _ in (0..num_neurons) {
            neurons.push(Neuron::new(num_weights, activation.clone()));
        }
        return Self {
            neurons
        };
    } 

    pub fn calculate_results(&mut self, vector: &Vec<f32>) -> Vec<f32> {
        let mut results = Vec::new();
        for neuron in  self.neurons.iter_mut() {
            let result = neuron.calculate(vector);
            results.push(result);
        } 
        return results;
    }

    pub fn get_neuron_count(&self) -> i32 {
        return self.neurons.len() as i32;
    } 

    pub fn perform_backpropagation(&mut self, inital_diffs: &mut Vec<f32>, learning_rate: f32) -> Vec<f32> {
        let mut sum_diff_with_respect_to_input_neurons = Vec::new();
        for (neuron , initial_diff) in zip(self.neurons.iter_mut(), inital_diffs.iter_mut()) {
             let activation_function_diff = neuron.calculate_diff_of_activation_function();
             *initial_diff *= activation_function_diff;
             neuron.subtract_diff_with_respect_to_weights_from_weights(*initial_diff, learning_rate);
             neuron.subtract_diff_with_respect_to_bias_from_bias(*initial_diff, learning_rate);
             let diff_with_respect_to_input_neurons = neuron.calculate_diff_with_respect_to_input_neurons(*initial_diff);
             if sum_diff_with_respect_to_input_neurons.len() == 0 {
                sum_diff_with_respect_to_input_neurons = diff_with_respect_to_input_neurons;
             } else {
                sum_diff_with_respect_to_input_neurons = zip(sum_diff_with_respect_to_input_neurons, diff_with_respect_to_input_neurons).map(|(sum, diff)| sum + diff).collect();
             }
        }
        sum_diff_with_respect_to_input_neurons
    }
}

pub struct LayerConfig {
    pub neuron_count: i32,
    pub activation: Activation
}


pub struct NeuralNetwork {
    layers: Vec<Layer>,
    embedding: Arc<Embedding>,
    tokenizer: Arc<Tokenizer>,
    learning_rate: f32
} 

impl NeuralNetwork {
    pub fn new(tokenizer: Arc<Tokenizer>, layers_configuration: &Vec<LayerConfig>, learning_rate: f32) -> Self {
        let mut layers: Vec<Layer> = Vec::new(); 
        let word_index = tokenizer.word_index();
        let embedding_dim = 50;  
        let embedding = Arc::new(Embedding::new(word_index.len() as i32, embedding_dim));
        let mut num_weights = embedding_dim;
        for layer_config in layers_configuration {
            let neuron_count = layer_config.neuron_count.clone();
            let layer = Layer::new(neuron_count.clone(), num_weights, layer_config.activation.clone());
            layers.push(layer);
            num_weights = neuron_count; 
        } 
        return Self {
            tokenizer,
            layers,  
            embedding,
            learning_rate
        }
    }

    pub fn train(&mut self, sentences: &Vec<String>, labels: &Vec<f32>) {
        let sequences = self.tokenizer.texts_to_sequences(&sentences); 
        for (sequence, label) in zip(sequences, labels) {
            let vector = self.embedding.pool_embedding(&sequence);  
            let value = self.run_iteration(&vector);
            println!("{}", value);
            self.perform_backpropagation(value, *label);
        }
    }

    pub fn perform_backpropagation(&mut self, observed: f32, label: f32) {
        let mut squared_error_diff = squared_error_diff(observed, label);
        let result_layer = self.layers.get(self.layers.len() - 1).unwrap();
        let mut initial_diffs: Vec<f32> = (0..result_layer.get_neuron_count()).map(|_| squared_error_diff.clone()).collect();
        let layers_len = self.layers.len();
        for i in (1..layers_len + 1) {
            let mut layer = self.layers.get_mut(layers_len - i).unwrap(); 
            initial_diffs = layer.perform_backpropagation(&mut initial_diffs, self.learning_rate.clone())
        }
    }

    fn run_iteration(&mut self, vector: &Vec<f32>) -> f32 {
        let mut current_input = vector.clone(); 
        for layer in self.layers.iter_mut() {
            let results = layer.calculate_results(&current_input);
            current_input = results;
        } 
        return current_input[0];
    }
} 