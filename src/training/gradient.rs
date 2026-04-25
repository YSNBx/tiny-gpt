pub fn gradient_descent(x_start: f32, learning_rate: f32, steps: i32) -> f32 {
  let mut x = x_start;
  for _i in 0..steps {
    let gradient: f32 = 2.0 * (x - 3.0) as f32;
    x = x - learning_rate * gradient;
    println!("new x {}", x);
  }
  x
}
