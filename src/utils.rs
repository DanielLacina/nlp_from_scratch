
pub fn sigmoid_diff(x: f32) -> f32 {
    sigmoid(x) * (1 as f32 - sigmoid(x))
}

pub fn sigmoid(x: f32) -> f32 {
    1.0/ (1.0 + (-x).exp())
}

pub fn relu_diff(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0) 
}