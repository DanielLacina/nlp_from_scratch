use core::num;
use std::{iter::zip, vec};
use std::sync::{Arc, RwLock};
use rand::Rng;
use std::thread;

use crate::{embedding::{Embedding}, tokenizer::Tokenizer, utils::*, layer::{Layer, LayerConfig}};


pub struct NeuralNetwork {
    layers: Arc<Vec<Layer>>,
    embedding: Arc<Embedding>,
    tokenizer: Arc<Tokenizer>,
    batch_size: i32,
    mse_scores: RwLock<Vec<f32>>
} 

impl NeuralNetwork {
    pub fn new(tokenizer: Arc<Tokenizer>, layers_configuration: &Vec<LayerConfig>, learning_rate: f32, batch_size: i32) -> Self {
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
            batch_size,
            layers: Arc::new(layers),  
            embedding,
            mse_scores: Default::default() 
        }
    }

    fn shuffle_data(&self, vectors: &mut Vec<Vec<f32>>, labels: &mut Vec<f32>) {
        let mut rng = rand::rng();
        for i in (0..vectors.len()).rev() {
            let j = rng.random_range(0..=i); 
            vectors.swap(i, j);
            labels.swap(i, j);
        } 
    }

    fn create_batches(&self, vectors: &Vec<Vec<f32>>, labels: &Vec<f32>) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let mut vectors = vectors.clone(); 
        let mut labels = labels.clone();
        self.shuffle_data(&mut vectors, &mut labels);
        let label_batches = labels.chunks(self.batch_size as usize).map(|chunk| chunk.to_vec()).collect(); 
        let vector_batches = vectors.chunks(self.batch_size as usize).map(|chunk| chunk.to_vec()).collect(); 
        return (vector_batches, label_batches);
    }

    pub fn train(&self, sentences: &Vec<String>, labels: &Vec<f32>, epochs: i32) {
        let sequences = self.tokenizer.texts_to_sequences(&sentences); 
        let vectors: Vec<Vec<f32>> = sequences.iter().map(|sequence| self.embedding.pool_embedding(&sequence)).collect();
        let mut mse_scores = self.mse_scores.write().unwrap();
        for _ in (0..epochs) {
            let (vector_batches, label_batches) = self.create_batches(&vectors, &labels); 
            let mut sum = 0.0;
            for (vector_batch, label_batch) in zip(vector_batches.iter(), label_batches.iter()) {
                let values = self.run_iterations(vector_batch);
                let squared_errors = zip(values.iter(), label_batch.iter()).map(|(result, label)| squared_error(*result, *label)).collect();
                let avg_squared_error = average_vector(&squared_errors);
                let avg_label = average_vector(label_batch);
                sum += avg_squared_error;
                self.perform_backpropagation(&values, &vec![avg_label]);
            }
            let mse = sum/vectors.len() as f32;
            mse_scores.push(mse);
        }
    }

    fn perform_backpropagation(&self, results: &Vec<f32>, labels: &Vec<f32>) {
        let mut input_gradients: Vec<f32> = zip(results, labels).map(|(result, label)| squared_error_diff(*result,  *label)).collect();
        for layer in self.layers.iter() {
            input_gradients = layer.perform_backpropagation(&input_gradients);
        }
    }

    pub fn run_iterations(&self, vector_batch: &Vec<Vec<f32>>) -> Vec<f32> {
        let handles = vector_batch.iter().map(|vector| {
            let layers = self.layers.clone();
            let vector = vector.clone();
            thread::spawn(move ||  {
               let results = Self::run_iteration(&vector, layers);
               return results[0];
            }       
            ) 
            }); 
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.join().unwrap(); 
            results.push(result);
        }  
        return results;
    }

    fn run_iteration(vector: &Vec<f32>, layers: Arc<Vec<Layer>>) -> Vec<f32> {
        let mut current_input = vector.clone();
        for layer in layers.iter() {
            let results = layer.calculate_results(&current_input);
            current_input = results;
        } 
        let results = current_input; 
        return results;
    }

    pub fn get_mse_scores(&self) -> Vec<f32> {
       return self.mse_scores.read().unwrap().clone(); 
    }
} 