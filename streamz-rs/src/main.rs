use std::fs;
use std::io::Write;
use std::path::Path;
use streamz_rs::{
    identify_speaker, load_wav_samples, train_from_files, SimpleNeuralNet, WINDOW_SIZE,
};


const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";

fn load_train_files(path: &str) -> Vec<(String, usize)> {
    if let Ok(content) = fs::read_to_string(path) {
        content
            .lines()
            .filter_map(|line| {
                let mut parts = line.split(',');
                if let (Some(p), Some(c)) = (parts.next(), parts.next()) {
                    c.trim().parse::<usize>().ok().map(|cls| (p.trim().into(), cls))
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![
            ("examples/training_data/arctic_a0008.wav".into(), 0),
            ("examples/training_data/arctic_a0015.wav".into(), 0),
            ("examples/training_data/arctic_a0021.wav".into(), 0),
            ("examples/training_data/arctic_b0196.wav".into(), 1),
            ("examples/training_data/arctic_b0356.wav".into(), 1),
            ("examples/training_data/arctic_b0417.wav".into(), 1),
        ]
    }
}

fn write_train_files(path: &str, files: &[(String, usize)]) {
    if let Ok(mut f) = std::fs::File::create(path) {
        for (p, c) in files {
            let _ = writeln!(f, "{},{}", p, c);
        }
    }
}

fn relabel_files(
    net: &SimpleNeuralNet,
    files: &mut [(String, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    for (path, class) in files.iter_mut() {
        let samples = load_wav_samples(path)?;
        let pred = identify_speaker(net, &samples);
        *class = pred;
    }
    Ok(())
}


/// Determine the number of speakers based on the provided training list.
/// The highest class index is assumed to be the last speaker and speaker
/// indexing starts at 0.
fn count_speakers(files: &[(String, usize)]) -> usize {
    files
        .iter()
        .map(|(_, class)| *class)
        .max()
        .map(|max| max + 1)
        .unwrap_or(0)
}

fn main() {
    let mut train_files = load_train_files(TRAIN_FILE_LIST);
    let num_speakers = count_speakers(&train_files);
    let train_refs: Vec<(&str, usize)> = train_files
        .iter()
        .map(|(p, c)| (p.as_str(), *c))
        .collect();

    let net = if Path::new(MODEL_PATH).exists() {
        match SimpleNeuralNet::load(MODEL_PATH) {
            Ok(n) => {
                println!("Loaded saved model from {}", MODEL_PATH);
                n
            }
            Err(e) => {
                eprintln!("Failed to load model: {}. Retraining...", e);
                let mut n = SimpleNeuralNet::new(WINDOW_SIZE, 32, num_speakers);
                if let Err(e) = train_from_files(&mut n, &train_refs, num_speakers, 30, 0.01) {
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
        if let Err(e) = train_from_files(&mut n, &train_refs, num_speakers, 30, 0.01) {
            eprintln!("Training failed: {}", e);
            return;
        }
        if let Err(e) = n.save(MODEL_PATH) {
            eprintln!("Failed to save model: {}", e);
        }
        n
    };

    if let Err(e) = relabel_files(&net, &mut train_files) {
        eprintln!("Failed to relabel files: {}", e);
    } else {
        write_train_files(TRAIN_FILE_LIST, &train_files);
        println!("Updated training file labels:");
        for (p, c) in &train_files {
            println!("{} -> speaker {}", p, c + 1);
        }
    }
}
