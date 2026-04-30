use crate::tensor::tensor::Tensor;
use crate::training::attention::Attention;
use crate::training::embedder::Embedder;
use crate::data::tokenizer::text_to_indices;

mod data;
mod tensor;
mod training;
mod util;

fn main() {
  let embedder = Embedder::new(26, 4);
  let indices = text_to_indices("hello");
  let forwarded: Tensor = embedder.forward(indices);

  let attention: Attention = Attention::new(4);
  let output = attention.forward(&forwarded);
  println!("{}", output);
}
