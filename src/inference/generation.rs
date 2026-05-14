use crate::data::tokenizer::text_to_indices;
use crate::model::Model;

pub fn generate_greedy(model: &Model, seed: &str, steps: usize) {
  let mut text = seed.to_string();

  for _ in 0..steps {
    let indices = text_to_indices(&text);
    let (probs, _) = model.forward(&indices);

    let rows = probs.shape[0];
    let vocab_size = probs.shape[1];

    let last_row = rows - 1;
    let start = last_row * vocab_size;

    let mut best_idx = 0;
    let mut best_prob = 0.0;

    for j in 0..vocab_size {
      let prob = probs.data[start + j];

      if prob > best_prob {
        best_prob = prob;
        best_idx = j;
      }
    }

    let predicted_char = index_to_char(best_idx);
    text.push(predicted_char);
  }

  println!("Generated: {}", text);
}

fn index_to_char(index: usize) -> char {
  if index == 26 {
    ' '
  } else {
    (b'a' + index as u8) as char
  }
}
