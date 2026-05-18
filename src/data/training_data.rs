use crate::data::tokenizer::text_to_indices;

pub fn build_training_data(input: &str, seq_len: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
	let input_indices = text_to_indices(input);
	let mut training_data: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();

	for i in 0..input_indices.len() {
		if i + 1 + seq_len > input_indices.len() {
			break;
		}

		let input = &input_indices[i..seq_len + i];
		let target = &input_indices[i + 1..seq_len + i + 1];

		training_data.push((input.to_vec(), target.to_vec()));
	}

	training_data
}
