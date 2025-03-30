use crate::utils::*; 
use rand::Rng;
use std::iter::zip;


pub struct Neuron {
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
        sum += self.bias;
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