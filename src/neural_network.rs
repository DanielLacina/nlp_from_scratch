use std::iter::zip;
use std::sync::Arc;
use crate::{embedding::{Embedding}, tokenizer::Tokenizer, utils::*, layer::{Layer, LayerConfig}};


pub struct NeuralNetwork {
    layers: Vec<Layer>,
    embedding: Arc<Embedding>,
    tokenizer: Arc<Tokenizer>,
    mse_scores: Vec<f32>
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
            let layer = Layer::new(neuron_count.clone(), num_weights, layer_config.activation.clone(), learning_rate);
            layers.push(layer);
            num_weights = neuron_count; 
        } 
        return Self {
            tokenizer,
            layers,  
            embedding,
            mse_scores: Vec::new()
        }
    }

    pub fn train(&mut self, sentences: &Vec<String>, labels: &Vec<f32>, epochs: i32) {
        let sequences = self.tokenizer.texts_to_sequences(&sentences); 
        let vectors: Vec<Vec<f32>> = sequences.iter().map(|sequence| self.embedding.pool_embedding(&sequence)).collect();
        for _ in (0..epochs) {
            let mut sum = 0.0;
            for (vector, label) in zip(vectors.iter(), labels.iter()) {
                let values = self.run_iteration(&vector);
                let squared_errors = zip(&values, &vec![label]).map(|(result, label)| squared_error(*result, **label)).collect();
                let squared_error = sum_vector(&squared_errors);
                sum += squared_error;
                self.perform_backpropagation(&values, &vec![*label]);
            }
            let mse = sum/vectors.len() as f32;
            self.mse_scores.push(mse);
        }
    }

    fn perform_backpropagation(&mut self, results: &Vec<f32>, labels: &Vec<f32>) {
        let mut input_gradients: Vec<f32> = zip(results, labels).map(|(result, label)| squared_error_diff(*result,  *label)).collect();
        for layer in self.layers.iter_mut().rev() {
            input_gradients = layer.perform_backpropagation(&input_gradients);
        }
    }

    fn run_iteration(&mut self, vector: &Vec<f32>) -> Vec<f32> {
        let mut current_input = vector.clone(); 
        for layer in self.layers.iter_mut() {
            let results = layer.calculate_results(&current_input);
            current_input = results;
        } 
        let results = current_input; 
        return results;
    }

    pub fn get_mse_scores(&self) -> &Vec<f32> {
       return &self.mse_scores; 
    }
} 