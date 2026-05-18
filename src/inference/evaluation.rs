use crate::data::tokenizer::text_to_indices;
use crate::model::Model;

pub fn inspect_predictions(model: &Model, input: &str) -> String {
	let mut predicted: Vec<char> = Vec::new();

	let input_indices = text_to_indices(input);
	let (final_probs, _) = model.forward(&input_indices);

	let rows = final_probs.shape[0];
	let vocab_size = final_probs.shape[1];

	for i in 0..rows {
		let start = i * vocab_size;

		let mut best_idx = 0;
		let mut best_prob = 0.0;

		for j in 0..vocab_size {
			let prob = final_probs.data[start + j];

			if prob > best_prob {
				best_prob = prob;
				best_idx = j;
			}
		}

		let predicted_char = index_to_char(best_idx);
		predicted.push(predicted_char);

		println!("Position {}: predicted '{}' ({:.1}%)", i, predicted_char, best_prob * 100.0);
	}

	let predicted_text = predicted.iter().collect::<String>();
	println!("input: {} -> output: {:?}", input, predicted_text);
	predicted_text
}

fn index_to_char(index: usize) -> char {
	if index == 26 { ' ' } else { (b'a' + index as u8) as char }
}
