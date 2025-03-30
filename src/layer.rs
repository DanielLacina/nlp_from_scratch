use crate::neuron::Neuron;
use crate::utils::*;
use std::iter::zip;


pub struct LayerConfig {
    pub neuron_count: i32,
    pub activation: Activation
}

pub struct Layer {
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

    pub fn perform_backpropagation(&mut self, inital_diffs: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        let mut sum_diff_with_respect_to_input_neurons = Vec::new();
        for (neuron , initial_diff) in zip(self.neurons.iter_mut(), inital_diffs.iter()) {
             let activation_function_diff = neuron.calculate_diff_of_activation_function();
             let current_diff = *initial_diff * activation_function_diff;
             neuron.subtract_diff_with_respect_to_weights_from_weights(current_diff, learning_rate);
             neuron.subtract_diff_with_respect_to_bias_from_bias(current_diff, learning_rate);
             let diff_with_respect_to_input_neurons = neuron.calculate_diff_with_respect_to_input_neurons(current_diff);
             if sum_diff_with_respect_to_input_neurons.len() == 0 {
                sum_diff_with_respect_to_input_neurons = diff_with_respect_to_input_neurons;
             } else {
                sum_diff_with_respect_to_input_neurons = zip(sum_diff_with_respect_to_input_neurons, diff_with_respect_to_input_neurons).map(|(sum, diff)| sum + diff).collect();
             }
        }
        sum_diff_with_respect_to_input_neurons
    }
}