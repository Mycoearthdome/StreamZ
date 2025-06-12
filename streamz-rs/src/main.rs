use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use streamz_rs::{
    identify_speaker, load_wav_samples, train_from_files, SimpleNeuralNet, WINDOW_SIZE,
};

include!(concat!(env!("OUT_DIR"), "/train_files.rs"));

const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";

fn write_train_files(path: &str, files: &[(&str, usize)]) {
    if let Ok(mut f) = std::fs::File::create(path) {
        for &(p, c) in files {
            let _ = writeln!(f, "{},{}", p, c);
        }
    }
}

fn rebuild_project() {
    if let Ok(mut child) = std::process::Command::new("cargo")
        .args(&["build", "--release"])
        .spawn()
    {
        let _ = child.wait();
    }
}

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
                write_train_files(TRAIN_FILE_LIST, &TRAIN_FILES);
                rebuild_project();
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
        write_train_files(TRAIN_FILE_LIST, &TRAIN_FILES);
        rebuild_project();
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
