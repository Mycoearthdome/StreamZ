use std::sync::{Arc, Mutex};
use streamz_rs::{live_mic_stream, SimpleNeuralNet};

fn main() {
    let net = Arc::new(Mutex::new(SimpleNeuralNet::new(1, 8, 1)));
    if let Err(e) = live_mic_stream(net) {
        eprintln!("Error: {}", e);
    }
}
