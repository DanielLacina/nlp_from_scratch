use crate::utils::*; 
use rand::Rng;
use std::iter::zip;


pub struct Neuron {
    weights: Vec<f32>,
    bias: f32, 
    activation: Activation,
    learning_rate: f32,
    last_input: Vec<f32>,
    activated_result: f32,
    unactivated_result: f32
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
            last_input: vec![],
            unactivated_result: 0.0,
            activated_result: 0.0,
            weights,
            activation,
            bias: 0.0
        };
    }

    pub fn calculate_result(&mut self, vector: &Vec<f32>) -> f32 {
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

    pub fn perform_backpropagation(&mut self, gradient_value: f32) -> Vec<f32> {
        // returns initial gradient with respect to input values  
        let activation_diff = self.diff_of_activation();
        let updated_gradient_value = activation_diff * gradient_value; 
        let updated_input_gradient =  self.weights.clone().iter().map(|weight| weight * updated_gradient_value).collect(); 
        self.weights = zip(&self.last_input, &self.weights).map(|(input, weight)| weight - (*input * updated_gradient_value * self.learning_rate)).collect();
        self.bias -= updated_gradient_value * self.learning_rate; 
        return updated_input_gradient;  
    }

    fn diff_of_activation(&self) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let num_weights = 5; 
        let activation = Activation::Sigmoid;
        let learning_rate = 0.1;
        let neuron = Neuron::new(num_weights, activation, learning_rate);
        assert!(matches!(neuron.activation, activation));
        assert!(neuron.bias == 0.0);
        assert!(neuron.weights.len() as i32 == num_weights);
        assert!(neuron.learning_rate == learning_rate);
    }
    #[test]
    fn test_calculate() {
        let vector =  vec![5.0, 8.0, 7.0, 4.0];
        let mut neuron = Neuron::new(vector.len() as i32, Activation::Relu, 0.1);
        neuron.calculate_result(&vector);         
        assert!(neuron.last_input == vector);
        let expected_unactivated_result = dot_product(&neuron.weights, &vector) + neuron.bias;
        assert!(neuron.unactivated_result == expected_unactivated_result);
        assert!(neuron.activated_result == neuron.apply_activation(expected_unactivated_result));
    }
    #[test]
    fn test_backpropagation() {
        let mut neuron = Neuron::new(5, Activation::Sigmoid, 0.1);
        let initial_bias = neuron.bias;
        let initial_weights = neuron.weights.clone();
        let input_gradient_value = 0.2;
        let updated_gradient_input = neuron.perform_backpropagation(input_gradient_value);
        let diff_of_activation = neuron.diff_of_activation();
        let updated_gradient_value = input_gradient_value * diff_of_activation;


        let expected_gradient_input: Vec<f32> = initial_weights.iter().map(|weight| weight * updated_gradient_value).collect();
        assert!(updated_gradient_input == expected_gradient_input);
        assert!(neuron.bias == (initial_bias - (updated_gradient_value * neuron.learning_rate)));
        let expected_weights: Vec<f32> = zip(&neuron.last_input, &initial_weights).map(|(input, weight)| weight - (input * updated_gradient_value *neuron.learning_rate)).collect();
        assert!(neuron.weights == expected_weights);
    }
}