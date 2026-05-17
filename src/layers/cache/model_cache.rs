use crate::layers::cache::feed_forward_cache::FeedForwardCache;
use crate::layers::cache::output_cache::OutputCache;

pub struct ModelCache {
	pub output_cache: OutputCache,
	pub feed_forward_cache: FeedForwardCache,
}

#[rustfmt::skip]
impl ModelCache {
	pub fn new(output_cache: OutputCache, feed_forward_cache: FeedForwardCache) -> Self {
		ModelCache { 
      output_cache,
      feed_forward_cache 
    }
	}
}
