use std::fs;
use std::io::Write;
use std::path::Path;
use streamz_rs::{
    audio_metadata, identify_speaker_with_threshold, load_audio_samples, pretrain_network,
    train_from_files, SimpleNeuralNet, WINDOW_SIZE,
};

const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";
const CONF_THRESHOLD: f32 = 0.6;

fn load_train_files(path: &str) -> Vec<(String, Option<usize>)> {
    if let Ok(content) = fs::read_to_string(path) {
        let mut files = Vec::new();
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
                        continue;
                    }
                }
                files.push((path, None));
            }
        }
        files
    } else {
        Vec::new()
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
    if train_files.is_empty() {
        eprintln!("{} is empty", TRAIN_FILE_LIST);
        return;
    }
    print_file_specs(&train_files);

    let mut num_speakers = count_speakers(&train_files);
    let mut net = if Path::new(MODEL_PATH).exists() {
        match SimpleNeuralNet::load(MODEL_PATH) {
            Ok(n) => {
                println!("Loaded saved model from {}", MODEL_PATH);
                n
            }
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                SimpleNeuralNet::new(WINDOW_SIZE, 32, num_speakers.max(1))
            }
        }
    } else {
        if num_speakers == 0 {
            num_speakers = 1;
            train_files[0].1 = Some(0);
        }
        let mut n = SimpleNeuralNet::new(WINDOW_SIZE, 32, num_speakers.max(1));
        let train_refs: Vec<(&str, usize)> = train_files
            .iter()
            .filter_map(|(p, c)| c.map(|cls| (p.as_str(), cls)))
            .collect();
        if !train_refs.is_empty() {
            let out_sz = n.output_size();
            if let Err(e) = train_from_files(&mut n, &train_refs, out_sz, 30, 0.01) {
                eprintln!("Training failed: {}", e);
            }
        }
        n
    };

    for (path, class) in train_files.iter_mut() {
        match load_audio_samples(path) {
            Ok(samples) => {
                if let Some(label) = *class {
                    let sz = net.output_size();
                    pretrain_network(&mut net, &samples, label, sz, 30, 0.01);
                } else if let Some(pred) = identify_speaker_with_threshold(&net, &samples, CONF_THRESHOLD) {
                    *class = Some(pred);
                    let sz = net.output_size();
                    pretrain_network(&mut net, &samples, pred, sz, 30, 0.01);
                } else {
                    net.add_output_class();
                    let new_label = net.output_size() - 1;
                    *class = Some(new_label);
                    let sz = net.output_size();
                    pretrain_network(&mut net, &samples, new_label, sz, 30, 0.01);
                }
            }
            Err(e) => eprintln!("Skipping {}: {}", path, e),
        }
    }

    if let Err(e) = net.save(MODEL_PATH) {
        eprintln!("Failed to save model: {}", e);
    }
    write_train_files(TRAIN_FILE_LIST, &train_files);
    println!("Updated training file labels:");
    for (p, c) in &train_files {
        match c {
            Some(cls) => println!("{} -> speaker {}", p, cls + 1),
            None => println!("{} -> speaker unknown", p),
        }
    }
}
