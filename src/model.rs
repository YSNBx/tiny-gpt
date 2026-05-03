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

  pub fn forward(&self, indices: Vec<usize>) -> Tensor {
    let embedded: Tensor = self.embedding.forward(indices);
    let attentioned: Tensor = self.attention.forward(&embedded);
    let feed_forwarded: Tensor = self.feed_forward.forward(&attentioned);
    let output = self.output.forward(&feed_forwarded);
    output
  }
}
