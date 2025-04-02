use nlp_from_scratch::{tokenizer::{Tokenizer}, neural_network::{NeuralNetwork}, layer::{LayerConfig}, utils::*};
use std::sync::Arc;
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::io::{BufRead, BufReader};
use criterion::{
    black_box,
    criterion_group,
    criterion_main,
    Criterion
};

fn training_benchmark() {
   let mut file = File::open("../Sarcasm_Headlines_Dataset.json").unwrap(); // Open the file
    let reader = BufReader::new(file);
    let mut sentences = Vec::new();
    let mut labels = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let json_data: Value = serde_json::from_str(&line).unwrap();
        sentences.push(json_data["headline"].to_string());
        labels.push(json_data["is_sarcastic"].as_f64().unwrap() as f32);
        if labels.len() == 20 {
            break ; 
        }
    }
    let mut tokenizer = Tokenizer::new(Some("<OV>".to_string())); 
    tokenizer.fit_on_texts(&sentences);
    let layers_config = vec![LayerConfig {
        activation: Activation::Relu,
        neuron_count:  30 
    }, LayerConfig {
        activation: Activation::Relu,
        neuron_count:  10 
    }, LayerConfig {
        activation: Activation::Sigmoid,
        neuron_count: 1
    }];  
    let mut neural_network = NeuralNetwork::new(Arc::new(tokenizer), &layers_config, 0.1, 64);
    neural_network.train(&sentences, &labels, 10 );
}

fn bench_functions(c: &mut Criterion) {
    c.bench_function(
        "training_benchmark", 
        |b| b.iter(|| training_benchmark())
    );
}

criterion_group!(benches, bench_functions);
criterion_main!(benches);