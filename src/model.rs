use crate::tensor::tensor::Tensor;
use crate::layers::embedding::Embedding;
use crate::layers::attention::Attention;
use crate::layers::feed_forward::FeedForward;
use crate::layers::output::Output;

pub struct Model {
  embedding: Embedding,
  attention: Attention,
  feed_forward: FeedForward,
  output: Output,
}

impl Model {
  pub fn new(vocab_size: usize, embedding_dim: usize)  -> Model {
    Model {
      embedding: Embedding::new(vocab_size, embedding_dim),
      attention: Attention::new(embedding_dim),
      feed_forward: FeedForward::new(embedding_dim),
      output: Output::new(vocab_size, embedding_dim),
    }
  }

  pub fn forward(&self, indices: &[usize]) -> (Tensor, Tensor) {
    let embedded: Tensor = self.embedding.forward(indices);
    // let attentioned: Tensor = self.attention.forward(&embedded);
    // let feed_forwarded: Tensor = self.feed_forward.forward(&attentioned);
    let probs = self.output.forward(&embedded);
    (probs, embedded)
  }

  pub fn backward(&mut self, d_logits: &Tensor, final_hidden: &Tensor, input_indices: &[usize], learning_rate: f32) {
    let d_w_out = final_hidden.transpose().matmul(&d_logits);
    let d_embedding = &d_logits.matmul(&self.output.get_w().transpose());

    self.embedding.apply_gradient(&d_embedding, &input_indices, learning_rate);
    self.output.apply_gradient(&d_w_out, learning_rate);
  }
}
