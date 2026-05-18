use crate::tensor::tensor::Tensor;

#[derive(Clone)]
pub struct FeedForwardCache {
	pub input: Tensor,
	pub pre_relu: Tensor,
	pub activated: Tensor,
	pub output: Tensor,
}

impl FeedForwardCache {
	pub fn new(input: Tensor, pre_relu: Tensor, activated: Tensor, output: Tensor) -> Self {
		FeedForwardCache { input, pre_relu, activated, output }
	}
}
