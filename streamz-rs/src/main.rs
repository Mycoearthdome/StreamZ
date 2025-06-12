use std::sync::{Arc, Mutex};
use std::path::Path;
use streamz_rs::{live_mic_stream, SimpleNeuralNet};

const NUM_SPEAKERS: usize = 2;

fn main() {
    let model_path = "model.npz";
    let net = if Path::new(model_path).exists() {
        SimpleNeuralNet::load(model_path).unwrap_or_else(|_| SimpleNeuralNet::new(1, 8, NUM_SPEAKERS))
    } else {
        SimpleNeuralNet::new(1, 8, NUM_SPEAKERS)
    };
    let net = Arc::new(Mutex::new(net));
    if let Err(e) = live_mic_stream(net.clone(), NUM_SPEAKERS) {
        eprintln!("Error: {}", e);
    }
}
