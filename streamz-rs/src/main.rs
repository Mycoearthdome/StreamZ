use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use streamz_rs::{
    compute_speaker_embeddings, identify_speaker_cosine, load_audio_samples, pretrain_network,
    train_from_files, SimpleNeuralNet, FEATURE_SIZE,
};

const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";
/// Confidence threshold for assigning a sample to an existing speaker.
/// Higher values make the program less eager to reuse a known speaker
/// and instead create a new one when confidence is low.
const DEFAULT_CONF_THRESHOLD: f32 = 0.8;
/// Number of training epochs for each file.
const TRAIN_EPOCHS: usize = 15;
/// Dropout probability used during training.
const DROPOUT_PROB: f32 = streamz_rs::DEFAULT_DROPOUT;

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
    let args: Vec<String> = env::args().collect();
    let mut conf_threshold = DEFAULT_CONF_THRESHOLD;
    let eval_mode = args.iter().any(|a| a == "--eval");
    if let Some(idx) = args.iter().position(|a| a == "--threshold") {
        if let Some(val) = args.get(idx + 1) {
            match val.parse::<f32>() {
                Ok(v) => conf_threshold = v,
                Err(_) => eprintln!(
                    "Invalid value for --threshold '{}', using default {}",
                    val, DEFAULT_CONF_THRESHOLD
                ),
            }
        } else {
            eprintln!(
                "Missing value for --threshold, using default {}",
                DEFAULT_CONF_THRESHOLD
            );
        }
    }

    let mut train_files = load_train_files(TRAIN_FILE_LIST);
    if train_files.is_empty() {
        eprintln!("{} is empty", TRAIN_FILE_LIST);
        return;
    }

    if eval_mode {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut labelled: Vec<(String, usize)> = train_files
            .iter()
            .filter_map(|(p, c)| c.map(|cls| (p.clone(), cls)))
            .collect();
        if labelled.is_empty() {
            eprintln!("No labelled data available for evaluation");
            return;
        }
        labelled.shuffle(&mut rng);
        let split = (labelled.len() as f32 * 0.2).ceil() as usize;
        let mut eval_set = labelled.split_off(labelled.len().saturating_sub(split));
        let train_refs: Vec<(&str, usize)> =
            labelled.iter().map(|(p, c)| (p.as_str(), *c)).collect();
        let mut net = SimpleNeuralNet::new(
            FEATURE_SIZE,
            256,
            128,
            count_speakers(&train_files).max(1),
        );
        if !train_refs.is_empty() {
            let out_sz = net.output_size();
            let _ = train_from_files(
                &mut net,
                &train_refs,
                train_refs.len(),
                out_sz,
                TRAIN_EPOCHS,
                0.01,
                DROPOUT_PROB,
            );
        }
        let embeds = compute_speaker_embeddings(&net).unwrap_or_default();
        let total = eval_set.len();
        let correct = eval_set
            .par_iter()
            .filter(|(path, class)| match load_audio_samples(path) {
                Ok(samples) => identify_speaker_cosine(&net, &embeds, &samples, conf_threshold)
                    .map(|pred| pred == *class)
                    .unwrap_or(false),
                Err(e) => {
                    eprintln!("Skipping {}: {}", path, e);
                    false
                }
            })
            .count();
        println!(
            "Eval accuracy: {}/{} ({:.2}%)",
            correct,
            total,
            correct as f32 * 100.0 / total as f32
        );
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
                SimpleNeuralNet::new(FEATURE_SIZE, 256, 128, num_speakers.max(1))
            }
        }
    } else {
        if num_speakers == 0 {
            num_speakers = 1;
            train_files[0].1 = Some(0);
        }
        let mut n = SimpleNeuralNet::new(FEATURE_SIZE, 256, 128, num_speakers.max(1));
        let train_refs: Vec<(&str, usize)> = train_files
            .iter()
            .filter_map(|(p, c)| c.map(|cls| (p.as_str(), cls)))
            .collect();
        if !train_refs.is_empty() {
            let out_sz = n.output_size();
            if let Err(e) = train_from_files(
                &mut n,
                &train_refs,
                train_files.len(),
                out_sz,
                TRAIN_EPOCHS,
                0.01,
                DROPOUT_PROB,
            ) {
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

    let preload: Vec<_> = train_files
        .par_iter()
        .map(|(path, class)| {
            load_audio_samples(path)
                .map(|samples| (path.clone(), *class, samples))
                .map_err(|e| e.to_string())
        })
        .collect();

    let mut speaker_embeds = compute_speaker_embeddings(&net).unwrap_or_default();

    for ((path, class), result) in train_files.iter_mut().zip(preload) {
        pb.set_message(path.to_string());
        match result {
            Ok((_, _, samples)) => {
                if let Some(label) = *class {
                    let sz = net.output_size();
                    pretrain_network(
                        &mut net,
                        &samples,
                        label,
                        sz,
                        TRAIN_EPOCHS,
                        0.01,
                        DROPOUT_PROB,
                    );
                    net.record_training_file(label, path);
                } else if let Some(pred) =
                    identify_speaker_cosine(&net, &speaker_embeds, &samples, conf_threshold)
                {
                    *class = Some(pred);
                    let sz = net.output_size();
                    pretrain_network(
                        &mut net,
                        &samples,
                        pred,
                        sz,
                        TRAIN_EPOCHS,
                        0.01,
                        DROPOUT_PROB,
                    );
                    net.record_training_file(pred, path);
                } else {
                    net.add_output_class();
                    let new_label = net.output_size() - 1;
                    *class = Some(new_label);
                    let sz = net.output_size();
                    pretrain_network(
                        &mut net,
                        &samples,
                        new_label,
                        sz,
                        TRAIN_EPOCHS,
                        0.01,
                        DROPOUT_PROB,
                    );
                    net.record_training_file(new_label, path);
                }
                speaker_embeds = compute_speaker_embeddings(&net).unwrap_or_default();
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
    let processed_speakers = count_speakers(&train_files);
    println!("Processed {} speakers in this batch.", processed_speakers);
}
