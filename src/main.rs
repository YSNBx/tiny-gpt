use crate::model::Model;
use crate::tensor::tensor::Tensor;
use crate::data::tokenizer::text_to_indices;

mod data;
mod tensor;
mod layers;
mod util;
mod model;

fn main() {
  let indices = text_to_indices("hello");
  let model = Model::new(26, 4);
  let output: Tensor = model.forward(indices);
  println!("{}", output);
}
