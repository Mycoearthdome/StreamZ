// The live microphone streaming example is disabled because the audio
// backend dependencies are not included.
use streamz_rs::{SimpleNeuralNet, WINDOW_SIZE};

fn main() {
    let _net = SimpleNeuralNet::new(WINDOW_SIZE, 32, 1);
    println!("live_mic_stream example is not available in this build.");
}
