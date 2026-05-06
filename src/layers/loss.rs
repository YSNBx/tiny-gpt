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
