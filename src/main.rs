use crate::tensor::tensor::Tensor;
use crate::training::attention::Attention;
use crate::training::embedder::Embedder;
use crate::data::tokenizer::text_to_indices;
use crate::training::feed_forward::{self, FeedForward};

mod data;
mod tensor;
mod training;
mod util;

fn main() {
  let indices = text_to_indices("hello");
  let embedder = Embedder::new(26, 4);
  let forwarded: Tensor = embedder.forward(indices);

  let attention: Attention = Attention::new(4);
  let output = attention.forward(&forwarded);

  let feed_forward = FeedForward::new(4);
  let activated = feed_forward.forward(&output);
  println!("{}", activated);
}
