use crate::neuron::Neuron;
use crate::utils::*;
use std::iter::zip;
use std::sync::Arc;

pub struct LayerConfig {
    pub neuron_count: i32,
    pub activation: Activation
}

pub struct Layer {
    neurons: Arc<Vec<Neuron>>,
}

impl Layer {
    pub fn new(num_neurons: i32, num_weights: i32, activation: Activation, learning_rate: f32) -> Self {
        let mut neurons = Vec::new();
        for _ in (0..num_neurons) {
            neurons.push(Neuron::new(num_weights, activation.clone(), learning_rate));
        }
        return Self {
            neurons: Arc::new(neurons)
        };
    } 

    pub fn calculate_results(&self, vector: &Vec<f32>) -> Vec<f32> {
        let mut results = Vec::new();
        for neuron in self.neurons.iter() {
            let result = neuron.calculate_result(vector);
            results.push(result);
        } 
        return results;
    }

    pub fn get_neuron_count(&self) -> i32 {
        return self.neurons.len() as i32;
    } 

    pub fn perform_backpropagation(&self, input_gradient: &Vec<f32>) -> Vec<f32> {
        let mut updated_gradient_values = Vec::new();
        for (neuron , gradient_value) in zip(self.neurons.iter(), input_gradient.iter()) {
             let updated_input_gradient =  neuron.perform_backpropagation(*gradient_value);
             if updated_gradient_values.len() == 0 {
                updated_gradient_values = updated_input_gradient;
             } else {
                updated_gradient_values = zip(&updated_gradient_values, &updated_input_gradient).map(|(sum, update)| sum + update).collect();
             }
        }
        updated_gradient_values
    }
}