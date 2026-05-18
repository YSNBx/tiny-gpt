use crate::layers::cache::attention_cache::AttentionCache;
use crate::tensor::tensor::Tensor;
use crate::util::vec_randomizer::random_vec;

#[derive(Debug)]
pub struct Attention {
	w_q: Tensor,
	w_k: Tensor,
	w_v: Tensor,
}

impl Attention {
	#[rustfmt::skip]
	pub fn new(embedding_dim: usize) -> Self {
		let size = embedding_dim * embedding_dim;
		Attention { 
			w_q: Tensor::new(random_vec(size), vec![embedding_dim, embedding_dim]),
			w_k: Tensor::new(random_vec(size), vec![embedding_dim, embedding_dim]),
			w_v: Tensor::new(random_vec(size), vec![embedding_dim, embedding_dim]) 
		}
	}

	pub fn forward(&self, input: &Tensor) -> (Tensor, AttentionCache) {
		let q = input.matmul(&self.w_q);
		let k = input.matmul(&self.w_k);
		let v = input.matmul(&self.w_v);

		let scores = q.matmul(&k.transpose());
		let scaled_scores = scores.divide_scalar((self.w_q.shape[0] as f32).sqrt());
		let weights = scaled_scores.softmax();
		let output = weights.matmul(&v);

		let attention_cache = AttentionCache::new(
		                                          input.clone(),
		                                          q.clone(),
		                                          k.clone(),
		                                          v.clone(),
		                                          scores.clone(),
		                                          scaled_scores.clone(),
		                                          weights.clone(),
		                                          output.clone(),
		);

		(output, attention_cache)
	}
}

impl std::fmt::Display for Attention {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Attention(w_q={:?}, w_k={:?}, w_v={:?})", self.w_q, self.w_k, self.w_v)
	}
}
