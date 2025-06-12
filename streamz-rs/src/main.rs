use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::Write;
use std::path::Path;
use streamz_rs::{
    identify_speaker_with_threshold, load_audio_samples, pretrain_network, train_from_files,
    SimpleNeuralNet, WINDOW_SIZE,
};

const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";
/// Confidence threshold for assigning a sample to an existing speaker.
/// Higher values make the program less eager to reuse a known speaker
/// and instead create a new one when confidence is low.
const CONF_THRESHOLD: f32 = 0.8;

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
            if let Err(e) =
                train_from_files(&mut n, &train_refs, train_files.len(), out_sz, 30, 0.01)
            {
                eprintln!("Training failed: {}", e);
            }
        }
        n
    };

    let pb = ProgressBar::new(train_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} ETA {eta}")
            .unwrap(),
    );

    for (path, class) in train_files.iter_mut() {
        pb.set_message(path.to_string());
        match load_audio_samples(path) {
            Ok(samples) => {
                if let Some(label) = *class {
                    let sz = net.output_size();
                    pretrain_network(&mut net, &samples, label, sz, 30, 0.01);
                    net.record_training_file(label, path);
                } else if let Some(pred) =
                    identify_speaker_with_threshold(&net, &samples, CONF_THRESHOLD)
                {
                    *class = Some(pred);
                    let sz = net.output_size();
                    pretrain_network(&mut net, &samples, pred, sz, 30, 0.01);
                    net.record_training_file(pred, path);
                } else {
                    net.add_output_class();
                    let new_label = net.output_size() - 1;
                    *class = Some(new_label);
                    let sz = net.output_size();
                    pretrain_network(&mut net, &samples, new_label, sz, 30, 0.01);
                    net.record_training_file(new_label, path);
                }
                if let Err(e) = net.save(MODEL_PATH) {
                    eprintln!("Failed to save model: {}", e);
                }
            }
            Err(e) => eprintln!("Skipping {}: {}", path, e),
        }
        pb.inc(1);
    }

    pb.finish_and_clear();

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
