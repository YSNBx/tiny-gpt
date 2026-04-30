#[derive(Clone, Debug)]
pub struct Tensor {
  pub data: Vec<f32>,
  pub shape: Vec<usize>
}

impl Tensor {
  pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
    let expected: usize = shape.iter().product();
    assert_eq!(data.len(), expected,
      "Data length {} doesn't match shape {:?} (expected {})",
      data.len(), shape, expected
    );
    Tensor { data , shape }
  }

  pub fn add(&self, other: &Tensor) -> Tensor {
    self.assert_equal(other);
    let new_data: Vec<f32> = self.data
      .iter()
      .zip(other.data.iter())
      .map(|(a, b)| a + b)
      .collect();
    Tensor { data: new_data, shape: self.shape.clone() }
  }

  pub fn multiply(&self, other: &Tensor) -> Tensor {
    self.assert_equal(other);
    let new_data: Vec<f32> = self.data
      .iter()
      .zip(other.data.iter())
      .map(|(a, b)| a * b)
      .collect();
    Tensor { data: new_data, shape: self.shape.clone() }
  }

  pub fn matmul(&self, other: &Tensor) -> Tensor {
    self.assert_equal_dimensions(other);
    let mut new: Vec<f32> = Vec::new();

    for i in 0..self.shape[0] {
      for j in 0..other.shape[1] {
        let mut sum = 0.0;
        for k in 0..self.shape[1] {
          sum += self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
        }
        new.push(sum);
      }
    }
    Tensor { data: new, shape: vec![self.shape[0], other.shape[1]] }
  }

  pub fn transpose(&self) -> Tensor {
    let mut v: Vec<f32> = Vec::new();

    for i in 0..self.shape[1] {
      for j in 0..self.shape[0] {
        v.push(self.data[j * self.shape[1] + i])
      }
    }
    Tensor { data: v, shape: vec![self.shape[1], self.shape[0]] }
  }

  pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
    assert_eq!(self.shape.iter().product::<usize>(), new_shape.iter().product::<usize>(),
      "Sum of new shape {:?} not equal to old shape {:?}", new_shape, self.shape,
    );
    Tensor { data: self.data.clone(), shape: new_shape }
  }

  pub fn softmax(&self) -> Tensor {
    let mut results: Vec<f32> = Vec::new();
    for i in 0..self.shape[0] {
      let start = i * self.shape[1];
      let end = start + self.shape[1];

      let slice = &self.data[start..end];
      let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
      let exp_sums: Vec<f32> = slice.iter()
        .map(|x| (*x - max).exp())
        .collect();
      let divisor: f32 = exp_sums.iter().sum();
      results.extend(exp_sums.iter().map(|x| *x / divisor));
    }
    Tensor { data: results, shape: self.shape.clone() }
  }

  pub fn divide_scalar(&self, scalar: f32) -> Tensor {
    let data = self.data
      .iter()
      .map(|x| *x / scalar)
      .collect();

    Tensor { data: data, shape: self.shape.clone() }
  }

  fn assert_equal(&self, other: &Tensor) {
    assert_eq!(&self.shape, &other.shape, 
      "Tensor shapes are not equal! {:?} != {:?}", &self.shape, other.shape
    );
  }
  
  fn assert_equal_dimensions(&self, other: &Tensor) {
    assert_eq!(self.shape[1],other.shape[0],
      "Inner dimensions must match: {} vs {}", self.shape[1], other.shape[0]
    );
  }
}

impl std::fmt::Display for Tensor {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Tensor(shape={:?}, data={:?})", self.shape, self.data)
  }
}
