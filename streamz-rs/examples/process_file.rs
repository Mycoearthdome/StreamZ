use streamz_rs::{process_wav, SimpleNeuralNet};

fn main() {
    let model = SimpleNeuralNet::new(16, 8, 16);
    let input = "input.wav";
    let output = "output.wav";
    if let Err(e) = process_wav(input, output, &model) {
        eprintln!("Error: {}", e);
    } else {
        println!("Processed file saved to {}", output);
    }
}
