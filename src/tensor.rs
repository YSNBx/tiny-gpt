pub struct Tensor {
  pub data: Vec<f32>,
  pub shape: Vec<usize>
}

impl Tensor {
  pub fn new (data: Vec<f32>, shape: Vec<usize>) -> Self {
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

  pub fn multiply_matrix(&self, other: &Tensor) -> Tensor {
    self.assert_equal_dimensions(other);
    let mut new: Vec<f32> = Vec::new();

    for i in 0..self.shape[0] { //[2,3]
      for j in 0..other.shape[1] { //[3, 2]
        let mut sum = 0.0;
        for k in 0..self.shape[1] {
          sum += self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
        }
        new.push(sum);
      }
    }
    Tensor { data: new, shape: vec![self.shape[0], other.shape[1]] }
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
