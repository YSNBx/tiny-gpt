use crate::util::vec_randomizer::random_vec;
use crate::tensor::tensor::Tensor;

pub struct Output {
  w: Tensor //weight matrix
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

  pub fn apply_gradient(&mut self, grad: &Tensor, learning_rate: f32) {
    assert_eq!(self.w.shape, grad.shape);
    for i in 0..self.w.data.len() {
      self.w.data[i] -= learning_rate * grad.data[i];
    }
  }

  pub fn get_w(&self) -> Tensor {
    self.w.clone()
  }
}

impl std::fmt::Display for Output {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Output(w={:?})", self.w)
  }
}
