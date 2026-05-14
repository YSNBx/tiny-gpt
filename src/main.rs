use std::char;

use crate::data::training_data::build_training_data;
use crate::inference::evaluation::inspect_predictions;
use crate::inference::generation::generate_greedy;
use crate::layers::loss::{self, cross_entropy_backward};
use crate::model::Model;
use crate::data::tokenizer::text_to_indices;

mod data;
mod tensor;
mod layers;
mod util;
mod model;
mod inference;

fn main() {
  let mut model: Model = Model::new(27, 8);
  let input = "the quick brown fox jumps over the lazy dog the fox and the dog are friends the quick fox runs and jumps the lazy dog sleeps on the mat the brown dog and the quick fox play in the park the dog runs after the fox the fox jumps over the log";

  let dataset: Vec<(Vec<usize>, Vec<usize>)> = build_training_data(input, 8);
  let learning_rate = 0.1;

  for epoch in 0..5000 {
    let mut total_loss = 0.0;

    for (input_indices, target_indices) in &dataset {
      let (probs, final_hidden) = model.forward(&input_indices);
      let loss = loss::cross_entropy_loss(&probs, &target_indices);
      total_loss += loss;

      let d_logits = cross_entropy_backward(&probs, &target_indices);
      model.backward(&d_logits, &final_hidden, input_indices, learning_rate);
    }

    if epoch % 200 == 0 {
      let avg_loss = total_loss / dataset.len() as f32;
      println!("Epoch {}: avg_loss = {}", epoch, avg_loss);
    }
  }
  inspect_predictions(&model, "the fox");
  generate_greedy(&model, "the", 30);
}
