use crate::config::TrainingConfig;
use crate::data::corpus;
use crate::data::training_data::build_training_data;
use crate::inference::evaluation::inspect_predictions;
use crate::inference::generation::generate_greedy;
use crate::layers::loss::{self, cross_entropy_backward};
use crate::model::Model;

mod config;
mod data;
mod inference;
mod layers;
mod model;
mod tensor;
mod util;

fn main() {
	let config = TrainingConfig::default();
	let mut model: Model = Model::new(config.vocab_size, config.embedding_dim);
	let training_data = build_training_data(corpus::TRAINING_TEXT, config.sequence_length);

	for epoch in 0..config.epochs {
		let mut total_loss = 0.0;

		for (context_indices, label_indices) in &training_data {
			let (probabilities, model_cache) = model.forward(context_indices);
			let loss = loss::cross_entropy_loss(&probabilities, label_indices);
			total_loss += loss;

			let d_logits = cross_entropy_backward(&probabilities, label_indices);
			model.backward(&d_logits, &model_cache, context_indices, config.learning_rate);
		}

		if epoch % config.log_every == 0 {
			let avg_loss = total_loss / training_data.len() as f32;
			println!("Epoch {}: avg_loss = {}", epoch, avg_loss);
		}
	}
	inspect_predictions(&model, "the fox");
	generate_greedy(&model, "the", config.generation_steps);
}
