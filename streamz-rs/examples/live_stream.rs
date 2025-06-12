use std::sync::{Arc, Mutex};
use streamz_rs::{live_mic_stream, SimpleNeuralNet, WINDOW_SIZE};

fn main() {
    let net = Arc::new(Mutex::new(SimpleNeuralNet::new(WINDOW_SIZE, 32, 1)));
    if let Err(e) = live_mic_stream(net, 1) {
        eprintln!("Error: {}", e);
    }
}
