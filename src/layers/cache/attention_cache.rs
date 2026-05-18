use crate::tensor::tensor::Tensor;

pub struct AttentionCache {
	pub input: Tensor,
	pub q: Tensor,
	pub k: Tensor,
	pub v: Tensor,
	pub scores: Tensor,
	pub scaled_scores: Tensor,
	pub weights: Tensor,
	pub output: Tensor,
}

impl AttentionCache {
	#[rustfmt::skip]
	#[allow(clippy::too_many_arguments)]
	pub fn new(input: Tensor, q: Tensor, k: Tensor, v: Tensor, 
						 scores: Tensor, scaled_scores: Tensor, weights: Tensor, output: Tensor) -> Self {
		AttentionCache { input, q, k, v, scores, scaled_scores, weights, output }
	}
}
