
use crate::{tensor::tensor::Tensor, util::vec_randomizer::random_vec};


pub struct Embedder {
  tensor: Tensor
}

impl Embedder {
  pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
    Embedder { tensor: Tensor::new(random_vec(vocab_size * embedding_dim), vec![vocab_size, embedding_dim]) }
  }

  pub fn forward(&self, indices: Vec<usize>) -> Tensor {
    let mut result: Vec<f32> = Vec::new();
    let dim = self.tensor.shape[1];

    for &i in &indices {
      let start = i * self.tensor.shape[1];
      let end = start + self.tensor.shape[1];
      result.extend_from_slice(&self.tensor.data[start..end]);
    }
    Tensor { data: result, shape: vec![indices.len(), dim] }
  }
}
