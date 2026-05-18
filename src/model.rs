use crate::layers::attention::Attention;
use crate::layers::cache::model_cache::ModelCache;
use crate::layers::embedding::Embedding;
use crate::layers::feed_forward::FeedForward;
use crate::layers::output::Output;
use crate::tensor::tensor::Tensor;

pub struct Model {
	embedding: Embedding,
	attention: Attention,
	feed_forward: FeedForward,
	output: Output,
}

impl Model {
	#[rustfmt::skip]
	pub fn new(vocab_size: usize, embedding_dim: usize) -> Model {
		Model { 
			embedding: Embedding::new(vocab_size, embedding_dim),
			attention: Attention::new(embedding_dim),
			feed_forward: FeedForward::new(embedding_dim),
			output: Output::new(vocab_size, embedding_dim) 
		}
	}

	pub fn forward(&self, indices: &[usize]) -> (Tensor, ModelCache) {
		let embedded: Tensor = self.embedding.forward(indices);
		// let attentioned: Tensor = self.attention.forward(&embedded);
		let (hidden_states, ff_cache) = self.feed_forward.forward(&embedded); //TODO replace with attentioned
		let (probabilities, output_cache) = self.output.forward(&hidden_states);

		let model_cache: ModelCache = ModelCache::new(output_cache.clone(), ff_cache.clone());
		(probabilities, model_cache)
	}

	pub fn backward(&mut self, d_logits: &Tensor, cache: &ModelCache, context_indices: &[usize], learning_rate: f32) {
		let d_ff_out = self.output.backward(d_logits, &cache.output_cache, learning_rate);
		let d_embedding = self.feed_forward.backward(&d_ff_out, &cache.feed_forward_cache, learning_rate);
		self.embedding.apply_gradient(&d_embedding, context_indices, learning_rate);
	}
}
