use std::io::{self, Write};
use streamz_rs::{
    identify_speaker, load_wav_samples, train_from_files, SimpleNeuralNet, WINDOW_SIZE,
};

const TRAIN_FILES: [(&str, usize); 6] = [
    ("examples/training_data/arctic_a0008.wav", 0),
    ("examples/training_data/arctic_a0015.wav", 0),
    ("examples/training_data/arctic_a0021.wav", 0),
    ("examples/training_data/arctic_b0196.wav", 1),
    ("examples/training_data/arctic_b0356.wav", 1),
    ("examples/training_data/arctic_b0417.wav", 1),
];

const NUM_SPEAKERS: usize = 2;

fn main() {
    let mut net = SimpleNeuralNet::new(WINDOW_SIZE, 32, NUM_SPEAKERS);
    // Train longer and with a slightly higher learning rate so the tiny
    // dataset actually influences the model weights.
    if let Err(e) = train_from_files(&mut net, &TRAIN_FILES, NUM_SPEAKERS, 30, 0.01) {
        eprintln!("Training failed: {}", e);
        return;
    }
    println!("Training complete. Choose a speaker file to test:");
    println!("[1] Speaker 1");
    println!("[2] Speaker 2");
    print!("> ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return;
    }
    let path = match input.trim() {
        "1" => "examples/training_data/arctic_a0008.wav",
        "2" => "examples/training_data/arctic_b0196.wav",
        _ => {
            println!("Invalid selection");
            return;
        }
    };
    match load_wav_samples(path) {
        Ok(samples) => {
            let speaker = identify_speaker(&net, &samples);
            println!("Model prediction: Speaker {}", speaker);
        }
        Err(e) => eprintln!("Failed to load test file: {}", e),
    }
}
