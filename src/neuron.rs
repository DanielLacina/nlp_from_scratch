use crate::utils::*; 
use crate::f32_wrapper::F32Wrapper; 
use rand::Rng;
use std::iter::zip;
use std::sync::{RwLock};
use dashmap::DashMap;


pub struct Neuron {
    weights: RwLock<Vec<f32>>,
    bias: RwLock<f32>, 
    activation: Activation,
    learning_rate: f32,
    last_inputs: RwLock<Vec<Vec<f32>>>,
    activated_results: DashMap<F32Wrapper, ()>,
    unactivated_results: DashMap<F32Wrapper, ()>
}

impl Neuron {
    pub fn new(num_weights: i32, activation: Activation, learning_rate: f32) -> Self {
        let mut rng = rand::rng(); 
        let mut weights = Vec::new();
        for _ in (0..num_weights) {
            weights.push(rng.gen_range(-0.5..0.5));
        }
        return Self {
            learning_rate,
            last_inputs: Default::default(),
            unactivated_results: DashMap::new(),
            activated_results: DashMap::new(),
            weights: RwLock::new(weights),
            activation,
            bias: Default::default() 
        };
    }

    pub fn calculate_result(&self, vector: &Vec<f32>) -> f32 {
        {
            let mut last_inputs = self.last_inputs.write().unwrap();
            last_inputs.push(vector.clone());
        }
        let mut sum = 0.0;
        let weights = self.weights.read().unwrap();
        for (value , weight ) in zip(vector, weights.iter()) {
            sum += value * weight;
        } 
        let bias = self.bias.read().unwrap();
        sum += *bias;
        self.unactivated_results.insert(F32Wrapper::new(sum), ());
        let result = self.apply_activation(sum); 
        self.activated_results.insert(F32Wrapper::new(result), ());
        return result;
    }

    pub fn get_weights_count(&self) -> i32 {
        return self.weights.read().unwrap().len() as i32;
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

    fn clear_iteration_data(&self) {
        self.unactivated_results.clear(); 
        self.unactivated_results.clear();
        self.last_inputs.write().unwrap().clear();
    }

    fn update_parameters(&self, gradient_value: f32) -> Vec<f32> {
        let unactivated_results: Vec<f32> = self.unactivated_results.iter().map(|pair| pair.key().inner()).collect();  
        let unactivated_result = sum_vector(&unactivated_results);
        let activation_diff = self.diff_of_activation(unactivated_result);
        let updated_gradient_value = activation_diff * gradient_value; 
        let mut weights = self.weights.write().unwrap();
        let updated_input_gradient =  weights.clone().iter().map(|weight| weight * updated_gradient_value).collect(); 
        let last_inputs = self.last_inputs.read().unwrap(); 
        let last_input = pool_2d_vector(&last_inputs); 
        *weights = zip(last_input.iter(), weights.iter()).map(|(input, weight)| weight - (*input * updated_gradient_value * self.learning_rate)).collect();
        let mut bias = self.bias.write().unwrap();
        *bias -= updated_gradient_value * self.learning_rate; 
        return updated_input_gradient;
    } 

    pub fn perform_backpropagation(&self, gradient_value: f32) -> Vec<f32> {
        // returns initial gradient with respect to input values  
        let updated_input_gradient = self.update_parameters(gradient_value); 
        self.clear_iteration_data();
        return updated_input_gradient;  
    }

    fn diff_of_activation(&self, unactivated_result: f32) -> f32 {
        let activation_diff = match &self.activation {
            Activation::Relu => {
                relu_diff(unactivated_result)
            }
            Activation::Sigmoid => {
                sigmoid_diff(unactivated_result)
            }
        };
        activation_diff
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_initialization() {
//         let num_weights = 5; 
//         let activation = Activation::Sigmoid;
//         let learning_rate = 0.1;
//         let neuron = Neuron::new(num_weights, activation, learning_rate);
//         assert!(matches!(neuron.activation, activation));
//         assert!(neuron.bias == 0.0);
//         assert!(neuron.weights.len() as i32 == num_weights);
//         assert!(neuron.learning_rate == learning_rate);
//     }
//     #[test]
//     fn test_calculate() {
//         let vector =  vec![5.0, 8.0, 7.0, 4.0];
//         let mut neuron = Neuron::new(vector.len() as i32, Activation::Relu, 0.1);
//         neuron.calculate_result(&vector);         
//         assert!(neuron.last_input == vector);
//         let expected_unactivated_result = dot_product(&neuron.weights, &vector) + neuron.bias;
//         assert!(neuron.unactivated_result == expected_unactivated_result);
//         assert!(neuron.activated_result == neuron.apply_activation(expected_unactivated_result));
//     }
//     #[test]
//     fn test_backpropagation() {
//         let mut neuron = Neuron::new(5, Activation::Sigmoid, 0.1);
//         let initial_bias = neuron.bias;
//         let initial_weights = neuron.weights.clone();
//         let input_gradient_value = 0.2;
//         let updated_gradient_input = neuron.perform_backpropagation(input_gradient_value);
//         let diff_of_activation = neuron.diff_of_activation();
//         let updated_gradient_value = input_gradient_value * diff_of_activation;


//         let expected_gradient_input: Vec<f32> = initial_weights.iter().map(|weight| weight * updated_gradient_value).collect();
//         assert!(updated_gradient_input == expected_gradient_input);
//         assert!(neuron.bias == (initial_bias - (updated_gradient_value * neuron.learning_rate)));
//         let expected_weights: Vec<f32> = zip(&neuron.last_input, &initial_weights).map(|(input, weight)| weight - (input * updated_gradient_value *neuron.learning_rate)).collect();
//         assert!(neuron.weights == expected_weights);
//     }
// }