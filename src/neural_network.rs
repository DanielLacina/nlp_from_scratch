use std::iter::zip;
use std::sync::Arc;
use crate::{embedding::{Embedding}, tokenizer::Tokenizer, utils::*, layer::{Layer, LayerConfig}};


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

    pub fn train(&mut self, sentences: &Vec<String>, labels: &Vec<f32>, epochs: i32) {
        let sequences = self.tokenizer.texts_to_sequences(&sentences); 
        let vectors: Vec<Vec<f32>> = sequences.iter().map(|sequence| self.embedding.pool_embedding(&sequence)).collect();
        for _ in (0..epochs) {
            for (vector, label) in zip(vectors.iter(), labels.iter()) {
                let value = self.run_iteration(&vector);
                self.perform_backpropagation(value, *label);
            }
        }
    }

    fn perform_backpropagation(&mut self, observed: f32, label: f32) {
        let mut squared_error_diff = squared_error_diff(observed, label);
        let result_layer = self.layers.get(self.layers.len() - 1).unwrap();
        let mut initial_diffs: Vec<f32> = (0..result_layer.get_neuron_count()).map(|_| squared_error_diff.clone()).collect();
        for layer in self.layers.iter_mut().rev() {
            initial_diffs = layer.perform_backpropagation(&initial_diffs, self.learning_rate.clone());
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