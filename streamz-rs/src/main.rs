use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
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

const MODEL_PATH: &str = "model.npz";

/// Determine the number of speakers based on the TRAIN_FILES array.
/// The highest class index is assumed to be the last speaker and
/// speaker indexing starts at 0.
fn count_speakers() -> usize {
    TRAIN_FILES
        .iter()
        .map(|&(_, class)| class)
        .max()
        .map(|max| max + 1)
        .unwrap_or(0)
}

fn main() {
    let num_speakers = count_speakers();
    let net = if Path::new(MODEL_PATH).exists() {
        match SimpleNeuralNet::load(MODEL_PATH) {
            Ok(n) => {
                println!("Loaded saved model from {}", MODEL_PATH);
                n
            }
            Err(e) => {
                eprintln!("Failed to load model: {}. Retraining...", e);
                let mut n = SimpleNeuralNet::new(WINDOW_SIZE, 32, num_speakers);
                if let Err(e) = train_from_files(&mut n, &TRAIN_FILES, num_speakers, 30, 0.01) {
                    eprintln!("Training failed: {}", e);
                    return;
                }
                if let Err(e) = n.save(MODEL_PATH) {
                    eprintln!("Failed to save model: {}", e);
                }
                n
            }
        }
    } else {
        let mut n = SimpleNeuralNet::new(WINDOW_SIZE, 32, num_speakers);
        if let Err(e) = train_from_files(&mut n, &TRAIN_FILES, num_speakers, 30, 0.01) {
            eprintln!("Training failed: {}", e);
            return;
        }
        if let Err(e) = n.save(MODEL_PATH) {
            eprintln!("Failed to save model: {}", e);
        }
        n
    };
    println!("Model ready. Choose a speaker file to test:");

    // Pick one example file per speaker for testing prompts
    let mut sample_for_speaker: Vec<Option<&str>> = vec![None; num_speakers];
    for &(path, class) in &TRAIN_FILES {
        if class < num_speakers && sample_for_speaker[class].is_none() {
            sample_for_speaker[class] = Some(path);
        }
    }
    for (i, sample) in sample_for_speaker.iter().enumerate() {
        if sample.is_some() {
            println!("[{}] Speaker {}", i + 1, i + 1);
        }
    }
    print!("> ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return;
    }
    let choice: usize = match input.trim().parse() {
        Ok(n) if n >= 1 && n <= num_speakers => n,
        _ => {
            println!("Invalid selection");
            return;
        }
    };
    let path = match sample_for_speaker[choice - 1] {
        Some(p) => p,
        None => {
            println!("No sample for selected speaker");
            return;
        }
    };
    match load_wav_samples(path) {
        Ok(samples) => {
            let speaker = identify_speaker(&net, &samples);
            println!("Model prediction: Speaker {}", speaker + 1);
            println!("Press ESC to exit...");
            if enable_raw_mode().is_ok() {
                loop {
                    if event::poll(Duration::from_millis(500)).unwrap_or(false) {
                        if let Event::Key(key) = event::read().unwrap() {
                            if key.code == KeyCode::Esc {
                                break;
                            }
                        }
                    }
                }
                let _ = disable_raw_mode();
            }
        }
        Err(e) => eprintln!("Failed to load test file: {}", e),
    }
}
