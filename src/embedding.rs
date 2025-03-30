use std::vec;

use rand::Rng;

pub struct Embedding {
    vocab_size: i32,
    embedding_dim: i32,
    matrix: Vec<Vec<f32>>
}

impl Embedding {
    pub fn new(vocab_size: i32, embedding_dim: i32) -> Self {
        let mut rng = rand::rng();
        let mut matrix = Vec::new();
        for _ in 0..vocab_size {
            let mut vector = Vec::new(); 
            for _ in 0..embedding_dim {
                vector.push(rng.gen_range(0.0..1.0));
            }
            matrix.push(vector);
        } 
        return Self {
            matrix,
            vocab_size,
            embedding_dim
        }
    } 

    pub fn get_matrix(&self) -> &Vec<Vec<f32>> {
        return &self.matrix;
    }

    pub fn pool_embedding(&self, ids: &Vec<i32>) -> Vec<f32> {
        let mut pooled_vector: Vec<f32> = (0..self.embedding_dim).map(|_| 0 as f32).collect();
        for id in ids {
            let vector = self.matrix.get(*id as usize).unwrap(); 
            for (i, coordinate) in vector.iter().enumerate() {
                pooled_vector[i] += coordinate;
            }
        } 
        pooled_vector = pooled_vector.iter().map(|coordinate| *coordinate/ids.len() as f32).collect();
        pooled_vector
    }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test] 
   fn test_embedding() {
     let mut embedding = Embedding::new(100, 10);
     println!("{:?}", embedding.pool_embedding(&vec![5, 2, 3]))
   }
}

