use rand::Rng;

pub struct Embedding {
    vocab_size: i32,
    embedding_dim: i32,
    pub embeddings: Vec<Vec<f32>>
}

impl Embedding {
    pub fn new(vocab_size: i32, embedding_dim: i32) -> Self {
        let mut rng = rand::rng();
        let mut embeddings = Vec::new();
        for _ in 0..vocab_size {
            let mut embedding = Vec::new(); 
            for _ in 0..embedding_dim {
                embedding.push(rng.random());
            }
            embeddings.push(embedding);
        } 
        return Self {
            embeddings,
            vocab_size,
            embedding_dim
        }
    } 

    pub fn pool_embeddings(&mut self, ids: &Vec<i32>) -> Vec<f32> {
        let mut pooled_embedding: Vec<f32> = (0..self.embedding_dim).map(|_| 0 as f32).collect();
        for id in ids {
            let embedding = self.embeddings.get(*id as usize).unwrap(); 
            for (i, coordinate) in embedding.iter().enumerate() {
                pooled_embedding[i] += coordinate;
            }
        } 
        pooled_embedding = pooled_embedding.iter().map(|coordinate| *coordinate/ids.len() as f32).collect();
        pooled_embedding
    }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test] 
   fn test_embedding() {
     let mut embedding = Embedding::new(100, 10);
     println!("{:?}", embedding.pool_embeddings(&vec![5, 2, 3]))
   }
}

