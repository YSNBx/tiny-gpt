use std::time::{SystemTime, UNIX_EPOCH};

const ARBITRARY: u128 = 0x5DEECE66D;

pub fn random_vec(size: usize) -> Vec<f32> {
  let mut seed = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .expect("time went backwards")
    .as_nanos();

  let mut values: Vec<f32> = Vec::with_capacity(size);

  for _ in 0..size {
    seed ^= ARBITRARY;
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed = seed.wrapping_mul(0x27d4eb2d);

    let val = (seed % 10_000) as f32 / 10_000.0;
    let val = val * 2.0 - 1.0;
    let val = (val * 100.0).round() / 100.0;

    values.push(val);
  }
  values
}
