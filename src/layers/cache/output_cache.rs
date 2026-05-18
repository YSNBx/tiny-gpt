use crate::tensor::tensor::Tensor;

#[derive(Clone)]
pub struct OutputCache {
	pub input: Tensor, // the hidden states passed into output layer
	pub logits: Tensor,
	pub probabilities: Tensor,
}

impl OutputCache {
	pub fn new(input: Tensor, logits: Tensor, probabilities: Tensor) -> Self {
		OutputCache { input, logits, probabilities }
	}
}
