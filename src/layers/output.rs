use crate::{tensor::tensor::Tensor, util::vec_randomizer::random_vec};

pub struct Output {
  w: Tensor
}

impl Output {
  pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
    Output {
      w: Tensor::new(random_vec(vocab_size * embedding_dim), vec![embedding_dim, vocab_size])
    }
  }

  pub fn forward(&self, input: &Tensor) -> Tensor {
    self.logits(input).softmax()
  }

  pub fn logits(&self, input: &Tensor) -> Tensor {
    input.matmul(&self.w)
  }
}

impl std::fmt::Display for Output {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Output(w={:?})", self.w)
  }
}
