use std::fs;
use std::io::Write;
use std::path::Path;
use streamz_rs::{
    audio_metadata, identify_speaker_with_threshold, load_audio_samples,
    train_from_files, SimpleNeuralNet, WINDOW_SIZE,
};

const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";
const CONF_THRESHOLD: f32 = 0.6;

fn load_train_files(path: &str) -> Vec<(String, Option<usize>)> {
    if let Ok(content) = fs::read_to_string(path) {
        let mut files = Vec::new();
        let mut has_labels = false;
        for line in content.lines() {
            let mut parts = line.split(',');
            if let Some(p) = parts.next() {
                let path = p.trim().to_string();
                if path.is_empty() {
                    continue;
                }
                if let Some(c) = parts.next() {
                    if let Ok(cls) = c.trim().parse::<usize>() {
                        files.push((path, Some(cls)));
                        has_labels = true;
                        continue;
                    }
                }
                files.push((path, None));
            }
        }
        if !has_labels {
            files.extend([
                ("examples/training_data/arctic_a0008.wav".into(), Some(0)),
                ("examples/training_data/arctic_a0015.wav".into(), Some(0)),
                ("examples/training_data/arctic_a0021.wav".into(), Some(0)),
                ("examples/training_data/arctic_b0196.wav".into(), Some(1)),
                ("examples/training_data/arctic_b0356.wav".into(), Some(1)),
                ("examples/training_data/arctic_b0417.wav".into(), Some(1)),
            ]);
        }
        files
    } else {
        vec![
            ("examples/training_data/arctic_a0008.wav".into(), Some(0)),
            ("examples/training_data/arctic_a0015.wav".into(), Some(0)),
            ("examples/training_data/arctic_a0021.wav".into(), Some(0)),
            ("examples/training_data/arctic_b0196.wav".into(), Some(1)),
            ("examples/training_data/arctic_b0356.wav".into(), Some(1)),
            ("examples/training_data/arctic_b0417.wav".into(), Some(1)),
        ]
    }
}

fn write_train_files(path: &str, files: &[(String, Option<usize>)]) {
    if let Ok(mut f) = std::fs::File::create(path) {
        for (p, c) in files {
            match c {
                Some(cls) => {
                    let _ = writeln!(f, "{},{}", p, cls);
                }
                None => {
                    let _ = writeln!(f, "{}", p);
                }
            }
        }
    }
}

fn print_file_specs(files: &[(String, Option<usize>)]) {
    println!("Detected file specs:");
    for (path, _) in files {
        match audio_metadata(path) {
            Ok((sr, bits)) => println!("{} -> {} Hz, {} bits", path, sr, bits),
            Err(e) => eprintln!("{} -> failed to read metadata: {}", path, e),
        }
    }
}

fn relabel_files(
    net: &SimpleNeuralNet,
    files: &mut [(String, Option<usize>)],
) -> Result<(), Box<dyn std::error::Error>> {
    for (path, class) in files.iter_mut() {
        let samples = load_audio_samples(path)?;
        if let Some(pred) = identify_speaker_with_threshold(net, &samples, CONF_THRESHOLD) {
            *class = Some(pred);
        }
    }
    Ok(())
}

/// Determine the number of speakers based on the provided training list.
/// The highest class index is assumed to be the last speaker and speaker
/// indexing starts at 0.
fn count_speakers(files: &[(String, Option<usize>)]) -> usize {
    files
        .iter()
        .filter_map(|(_, class)| *class)
        .max()
        .map(|max| max + 1)
        .unwrap_or(0)
}

fn main() {
    let mut train_files = load_train_files(TRAIN_FILE_LIST);
    print_file_specs(&train_files);
    let num_speakers = count_speakers(&train_files);
    let train_refs: Vec<(&str, usize)> = train_files
        .iter()
        .filter_map(|(p, c)| c.map(|cls| (p.as_str(), cls)))
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
        if train_refs.is_empty() {
            eprintln!(
                "No labeled training data found in {} and no saved model present.",
                TRAIN_FILE_LIST
            );
            return;
        }
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
            match c {
                Some(cls) => println!("{} -> speaker {}", p, cls + 1),
                None => println!("{} -> speaker unknown", p),
            }
        }
    }
}
