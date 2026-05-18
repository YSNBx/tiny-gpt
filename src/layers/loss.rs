use crate::tensor::tensor::Tensor;

pub fn cross_entropy_loss(output: &Tensor, targets: &[usize]) -> f32 {
	let input_size = targets.len();
	let vocab_size: usize = output.shape[1];
	let mut total_loss: f32 = 0.0;

	for row in 0..input_size {
		let target = targets[row];
		let prob = output.data[row * vocab_size + target].max(1e-9);
		total_loss += -prob.ln();
	}
	total_loss / input_size as f32
}

pub fn cross_entropy_backward(probs: &Tensor, targets: &[usize]) -> Tensor {
	let rows = targets.len();
	let vocab_size = probs.shape[1];

	let mut gradient = probs.data.clone();

	for row in 0..rows {
		let target = targets[row];
		let flat_index = row * vocab_size + target;

		gradient[flat_index] -= 1.0;
	}

	for value in gradient.iter_mut() {
		*value /= rows as f32;
	}

	Tensor { data: gradient, shape: probs.shape.clone() }
}
