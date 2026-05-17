use crate::{
	layers::cache::feed_forward_cache::FeedForwardCache, tensor::tensor::Tensor,
	util::vec_randomizer::random_vec,
};

pub struct FeedForward {
	w1: Tensor,
	w2: Tensor,
}

#[rustfmt::skip]
impl FeedForward {
	pub fn new(dim: usize) -> Self {
		let hidden_dim = dim * 4;
		FeedForward { 
      w1: Tensor::new(random_vec(dim * hidden_dim), vec![dim, hidden_dim]),
      w2: Tensor::new(random_vec(hidden_dim * dim), vec![hidden_dim, dim]) 
    }
	}

	pub fn forward(&self, input: &Tensor) -> (Tensor, FeedForwardCache) {
		let pre_relu = input.matmul(&self.w1);
		let activated = pre_relu.relu();
		let output = activated.matmul(&self.w2);
		let cache = FeedForwardCache::new(input.clone(), pre_relu.clone(), activated.clone(), output.clone());
		(output, cache)
	}

	pub fn backward(&mut self, grad: &Tensor, ff_cache: &FeedForwardCache, learning_rate: f32) {
    todo!()
  }
}
