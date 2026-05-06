use crate::layers::loss;
use crate::model::Model;
use crate::data::tokenizer::text_to_indices;

mod data;
mod tensor;
mod layers;
mod util;
mod model;

fn main() {
  let model: Model = Model::new(26, 4);

  let input_indices: Vec<usize> = text_to_indices("hell");
  let target_indices: Vec<usize> = text_to_indices("ello");

  let output = model.forward(input_indices);
  let loss = loss::cross_entropy_loss(&output, &target_indices);

  println!("{}", loss);
}
