use crate::training::gradient::gradient_descent;

mod tensor;
mod training;

fn main() {
    println!("{}", gradient_descent(0.0, 0.1, 50));
  // let _g: Gradient = Gradient { minimizer, learning_rate, steps };
  //
  // let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
  // println!("{}", t1);
  // let t2 = Tensor::new(vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5], vec![2, 3]).add(&t1);
  // println!("{}", t2);
  // let t3 = Tensor::new(vec![2.1, 2.2, 2.3, 2.4, 2.5, 2.6], vec![2, 3]).multiply(&t2);
  // println!("{}", t3);
  //
  // let t4 = &t1.multiply_matrix(&Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]));
  // println!("{}", t4);
  //
  // let t5 = &t1.transpose();
  // println!("{}", t5);
  //
  // let t6 = &t1.reshape(vec![3, 2]);
  // println!("{}", t6);
}
