pub struct TrainingConfig {
  pub vocab_size: usize,
  pub embedding_dim: usize,
  pub sequence_length: usize,
  pub learning_rate: f32,
  pub epochs: usize,
  pub log_every: usize,
  pub generation_steps: usize,
}

impl TrainingConfig {
  pub fn default() -> Self {
    TrainingConfig {
      vocab_size: 27,
      embedding_dim: 8,
      sequence_length: 8,
      learning_rate: 0.1,
      epochs: 5000,
      log_every: 200,
      generation_steps: 30,
    }
  }
}
