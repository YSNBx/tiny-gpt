use crate::{tensor::tensor::Tensor, util::vec_randomizer::random_vec};

#[derive(Debug)]
pub struct Attention {
  w_q: Tensor,
  w_k: Tensor,
  w_v: Tensor
}

impl Attention {
  pub fn new(embedding_dim: usize) -> Self {
    let size = embedding_dim * embedding_dim;
    Attention { 
      w_q: Tensor::new(random_vec(size), vec![embedding_dim, embedding_dim]),
      w_k: Tensor::new(random_vec(size), vec![embedding_dim, embedding_dim]),
      w_v: Tensor::new(random_vec(size), vec![embedding_dim, embedding_dim])
    }
  }

  pub fn forward(&self, input: &Tensor) -> Tensor {
    let q: Tensor = input.matmul(&self.w_q);
    let k: Tensor = input.matmul(&self.w_k);
    let v: Tensor = input.matmul(&self.w_v);
    let scores: Tensor = q.matmul(&k.transpose());
    let scaled_scores: Tensor = scores.divide_scalar((self.w_q.shape[0] as f32).sqrt());
    let weights: Tensor = scaled_scores.softmax();
    weights.matmul(&v)
  }
  
  pub fn compute_q(&self, input: &Tensor) -> Tensor {
    Tensor::matmul(input, &self.w_q)
  }

  pub fn compute_k(&self, input: &Tensor) -> Tensor {
    Tensor::matmul(input, &self.w_k)
  }

  pub fn compute_v(&self, input: &Tensor) -> Tensor {
    Tensor::matmul(input, &self.w_v)
  }

  pub fn compute_scores(&self, q: &Tensor, k: &Tensor) -> Tensor {
    let k_t = k.transpose();
    q.matmul(&k_t)
  }

  pub fn compute_output(&self, weights: &Tensor, v: &Tensor) -> Tensor {
    weights.matmul(v)
  }
}

impl std::fmt::Display for Attention {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Attention(w_q={:?}, w_k={:?}, w_v={:?})", self.w_q, self.w_k, self.w_v)
  }
}
