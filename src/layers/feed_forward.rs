use crate::{tensor::tensor::Tensor, util::vec_randomizer::random_vec};

pub struct FeedForward {
  w1: Tensor,
  w2: Tensor,
}

impl FeedForward {
  pub fn new(dim: usize) -> Self {
    let hidden_dim = dim * 4;
    FeedForward {
      w1: Tensor::new(random_vec(dim * hidden_dim), vec![dim, hidden_dim]),
      w2: Tensor::new(random_vec(hidden_dim * dim), vec![hidden_dim, dim])
    }
  } 

  pub fn forward(&self, input: &Tensor) -> Tensor {
    let hidden = input.matmul(&self.w1);
    let activated = hidden.relu();
    activated.matmul(&self.w2)
  }
}
