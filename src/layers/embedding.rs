
use crate::{tensor::tensor::Tensor, util::vec_randomizer::random_vec};

pub struct Embedding {
  tensor: Tensor
}

impl Embedding {
  pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
    Embedding { tensor: Tensor::new(random_vec(vocab_size * embedding_dim), vec![vocab_size, embedding_dim]) }
  }

  pub fn forward(&self, indices: &[usize]) -> Tensor {
    let mut result: Vec<f32> = Vec::new();
    let dim = self.tensor.shape[1];

    for i in indices {
      let start = i * self.tensor.shape[1];
      let end = start + self.tensor.shape[1];
      result.extend_from_slice(&self.tensor.data[start..end]);
    }
    Tensor { data: result, shape: vec![indices.len(), dim] }
  }

  pub fn apply_gradient(&mut self, d_embedding: &Tensor, input_indices: &[usize], learning_rate: f32) {
    for i in 0..input_indices.len() {
      let index_to_update = input_indices[i];

      let d_row_start = i * d_embedding.shape[1];
      let d_row_end = d_row_start + d_embedding.shape[1];
      let d_embedding_data = &d_embedding.data[d_row_start..d_row_end];

      let tensor_start = index_to_update * self.tensor.shape[1];

      for j in 0..self.tensor.shape[1] {
        self.tensor.data[j + tensor_start] -= learning_rate * d_embedding_data[j];
      }
    }
  }
}
