use std::hash::{Hash, Hasher};


#[derive(Clone, Copy, Debug)]
pub struct F32Wrapper(f32);

impl F32Wrapper {
    pub fn new(value: f32) -> Self {
        F32Wrapper(value)
    }

    fn approx_eq(self, other: F32Wrapper, tolerance: f32) -> bool {
        (self.0 - other.0).abs() < tolerance
    }

    pub fn inner(&self) -> f32 {
        return self.0;
    }
}

impl PartialEq for F32Wrapper {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(*other, 1e-5)  
    }
}

impl Eq for F32Wrapper {}

impl Hash for F32Wrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}