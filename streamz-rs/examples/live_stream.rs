// The live microphone streaming example is disabled because the audio
// backend dependencies are not included.
use streamz_rs::{SimpleNeuralNet, FEATURE_SIZE};

fn main() {
    // new network now expects two hidden layer sizes
    let _net = SimpleNeuralNet::new(FEATURE_SIZE, 32, 16, 1);
    println!("live_mic_stream example is not available in this build.");
}
