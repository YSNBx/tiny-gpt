use crate::layers::cache::feed_forward_cache::FeedForwardCache;
use crate::tensor::tensor::Tensor;
use crate::util::vec_randomizer::random_vec;

pub struct FeedForward {
	w1: Tensor,
	w2: Tensor,
}

impl FeedForward {
	#[rustfmt::skip]
	pub fn new(dim: usize) -> Self {
		let hidden_dim = dim * 4;
		FeedForward {
			w1: Tensor::new(random_vec(dim * hidden_dim), vec![dim, hidden_dim]),
			w2: Tensor::new(random_vec(hidden_dim * dim), vec![hidden_dim, dim]) 
		}
	}

	pub fn forward(&self, input: &Tensor) -> (Tensor, FeedForwardCache) {
		let pre_relu = input.matmul(&self.w1);
		let activated = pre_relu.relu_forward();
		let output = activated.matmul(&self.w2);
		let cache = FeedForwardCache::new(input.clone(), pre_relu.clone(), activated.clone(), output.clone());
		(output, cache)
	}

	pub fn backward(&mut self, grad: &Tensor, ff_cache: &FeedForwardCache, learning_rate: f32) -> Tensor {
		let activated = &ff_cache.activated;
		let d_w2 = activated.transpose().matmul(grad);
		let d_activated = grad.matmul(&self.w2.transpose());
		let d_pre_relu = d_activated.relu_backward(&ff_cache.pre_relu);
		let d_w1 = ff_cache.input.transpose().matmul(&d_pre_relu);
		let d_input = d_pre_relu.matmul(&self.w1.transpose());

		self.apply_gradient_w2(&d_w2, learning_rate);
		self.apply_gradient_w1(&d_w1, learning_rate);
		d_input
	}

	pub fn apply_gradient_w1(&mut self, d_w: &Tensor, learning_rate: f32) {
		assert_eq!(self.w1.shape, d_w.shape);
		for i in 0..d_w.data.len() {
			self.w1.data[i] -= learning_rate * d_w.data[i];
		}
	}

	pub fn apply_gradient_w2(&mut self, d_w: &Tensor, learning_rate: f32) {
		assert_eq!(self.w2.shape, d_w.shape);
		for i in 0..d_w.data.len() {
			self.w2.data[i] -= learning_rate * d_w.data[i];
		}
	}
}
