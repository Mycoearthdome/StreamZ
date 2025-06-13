use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use streamz_rs::{
    compute_speaker_embeddings, identify_speaker_cosine, identify_speaker_with_threshold,
    load_audio_samples, pretrain_network, train_from_files, SimpleNeuralNet, FEATURE_SIZE,
};

const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";
/// Confidence threshold for assigning a sample to an existing speaker.
/// Higher values make the program less eager to reuse a known speaker
/// and instead create a new one when confidence is low.
const DEFAULT_CONF_THRESHOLD: f32 = 0.8;
/// Fraction of the dataset used for the early burn-in phase when no
/// value is provided via `--burn-in-limit`.
const DEFAULT_BURN_IN_FRAC: f32 = 0.2;
/// Number of training epochs for each file.
const TRAIN_EPOCHS: usize = 10;
/// Dropout probability used during training.
const DROPOUT_PROB: f32 = streamz_rs::DEFAULT_DROPOUT;
/// Number of feature windows per training batch.
const BATCH_SIZE: usize = 8;

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
    let mut eval_split = 0.2f32;
    let mut burn_in_limit: Option<usize> = None;
    let mut max_speakers: Option<usize> = None;
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
    if let Some(idx) = args.iter().position(|a| a == "--eval-split") {
        if let Some(val) = args.get(idx + 1) {
            match val.parse::<f32>() {
                Ok(v) => eval_split = v.clamp(0.0, 1.0),
                Err(_) => eprintln!(
                    "Invalid value for --eval-split '{}', using default 0.2",
                    val
                ),
            }
        } else {
            eprintln!("Missing value for --eval-split, using default 0.2");
        }
    }
    if let Some(idx) = args.iter().position(|a| a == "--burn-in-limit") {
        if let Some(val) = args.get(idx + 1) {
            match val.parse::<usize>() {
                Ok(v) => burn_in_limit = Some(v),
                Err(_) => eprintln!(
                    "Invalid value for --burn-in-limit '{}', using automatic setting",
                    val
                ),
            }
        } else {
            eprintln!("Missing value for --burn-in-limit, using automatic setting");
        }
    }
    if let Some(idx) = args.iter().position(|a| a == "--max-speakers") {
        if let Some(val) = args.get(idx + 1) {
            match val.parse::<usize>() {
                Ok(v) => max_speakers = Some(v),
                Err(_) => eprintln!(
                    "Invalid value for --max-speakers '{}', using automatic setting",
                    val
                ),
            }
        } else {
            eprintln!("Missing value for --max-speakers, using automatic setting");
        }
    }

    let mut train_files = load_train_files(TRAIN_FILE_LIST);
    if train_files.is_empty() {
        eprintln!("{} is empty", TRAIN_FILE_LIST);
        return;
    }

    let dataset_size = train_files.len();
    let burn_in_default = ((dataset_size as f32) * DEFAULT_BURN_IN_FRAC).ceil() as usize;
    let _burn_in_limit = burn_in_limit.unwrap_or_else(|| burn_in_default.clamp(10, 50));
    let _max_speakers = max_speakers.unwrap_or_else(|| count_speakers(&train_files) + 10);

    if eval_mode {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let labelled: Vec<(String, usize)> = train_files
            .iter()
            .filter_map(|(p, c)| c.map(|cls| (p.clone(), cls)))
            .collect();
        if labelled.is_empty() {
            eprintln!("No labelled data available for evaluation");
            return;
        }
        use std::collections::HashMap;
        let mut by_class: HashMap<usize, Vec<String>> = HashMap::new();
        for (path, cls) in labelled {
            by_class.entry(cls).or_default().push(path);
        }
        let mut eval_set: Vec<(String, usize)> = Vec::new();
        let mut train_refs_owned: Vec<(String, usize)> = Vec::new();
        for (cls, mut paths) in by_class {
            paths.shuffle(&mut rng);
            let split_count = (paths.len() as f32 * eval_split).ceil() as usize;
            let eval_part: Vec<(String, usize)> = paths
                .split_off(paths.len().saturating_sub(split_count))
                .into_iter()
                .map(|p| (p, cls))
                .collect();
            let train_part: Vec<(String, usize)> = paths.into_iter().map(|p| (p, cls)).collect();
            eval_set.extend(eval_part);
            train_refs_owned.extend(train_part);
        }
        let train_refs: Vec<(&str, usize)> = train_refs_owned
            .iter()
            .map(|(p, c)| (p.as_str(), *c))
            .collect();
        let mut net =
            SimpleNeuralNet::new(FEATURE_SIZE, 512, 256, count_speakers(&train_files).max(1));
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
                BATCH_SIZE,
            );
        }
        let total = eval_set.len();
        let mut correct = 0usize;
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fnc = 0usize;
        for (path, class) in &eval_set {
            match load_audio_samples(path) {
                Ok(samples) => {
                    match identify_speaker_with_threshold(&net, &samples, conf_threshold) {
                        Some(pred) => {
                            if pred == *class {
                                tp += 1;
                                correct += 1;
                            } else {
                                fp += 1;
                                fnc += 1;
                            }
                        }
                        None => {
                            fnc += 1;
                        }
                    }
                }
                Err(e) => eprintln!("Skipping {}: {}", path, e),
            }
        }
        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if tp + fnc > 0 {
            tp as f32 / (tp + fnc) as f32
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        println!(
            "Eval accuracy: {}/{} ({:.2}%)",
            correct,
            total,
            correct as f32 * 100.0 / total as f32
        );
        println!(
            "Precision: {:.2}% Recall: {:.2}% F1: {:.2}%",
            precision * 100.0,
            recall * 100.0,
            f1 * 100.0
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
                SimpleNeuralNet::new(FEATURE_SIZE, 512, 256, num_speakers.max(1))
            }
        }
    } else {
        if num_speakers == 0 {
            num_speakers = 1;
            train_files[0].1 = Some(0);
        }
        let mut n = SimpleNeuralNet::new(FEATURE_SIZE, 512, 256, num_speakers.max(1));
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
                BATCH_SIZE,
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

    let mut total_loss = 0.0f32;
    let mut loss_count = 0usize;
    let mut embeddings = compute_speaker_embeddings(&net).unwrap_or_default();
    let mut update_embeddings = true;
    let mut _processed = 0;

    for (path, class) in train_files.iter_mut() {
        pb.set_message(path.to_string());

        match load_audio_samples(path) {
            Ok(samples) => {
                let dynamic_threshold = if loss_count < 50 {
                    0.95
                } else {
                    conf_threshold
                };

                if let Some(label) = *class {
                    // Known speaker: supervised training
                    let sz = net.output_size();
                    let loss = pretrain_network(
                        &mut net,
                        &samples,
                        label,
                        sz,
                        5, // Reduced TRAIN_EPOCHS for speed
                        0.01,
                        DROPOUT_PROB,
                        BATCH_SIZE,
                    );
                    total_loss += loss;
                    loss_count += 1;
                    net.record_training_file(label, path);
                    update_embeddings = true;
                } else {
                    // Unlabelled: try to match known speaker
                    if update_embeddings || embeddings.is_empty() {
                        embeddings = compute_speaker_embeddings(&net).unwrap_or_default();
                        update_embeddings = false;
                    }

                    if let Some(pred) =
                        identify_speaker_cosine(&net, &embeddings, &samples, dynamic_threshold)
                    {
                        *class = Some(pred);
                        let sz = net.output_size();
                        let loss = pretrain_network(
                            &mut net,
                            &samples,
                            pred,
                            sz,
                            5,
                            0.01,
                            DROPOUT_PROB,
                            BATCH_SIZE,
                        );
                        total_loss += loss;
                        loss_count += 1;
                        net.record_training_file(pred, path);
                        update_embeddings = true;
                    } else {
                        // New speaker: expand class
                        net.add_output_class();
                        let new_label = net.output_size() - 1;
                        *class = Some(new_label);
                        let sz = net.output_size();
                        let loss = pretrain_network(
                            &mut net,
                            &samples,
                            new_label,
                            sz,
                            5,
                            0.01,
                            DROPOUT_PROB,
                            BATCH_SIZE,
                        );
                        total_loss += loss;
                        loss_count += 1;
                        net.record_training_file(new_label, path);
                        update_embeddings = true;
                    }
                }

                // Save periodically
                if loss_count % 10 == 0 {
                    if let Err(e) = net.save(MODEL_PATH) {
                        eprintln!("Failed to save model: {}", e);
                    }
                    embeddings = compute_speaker_embeddings(&net).unwrap_or_default();
                }
            }
            Err(e) => eprintln!("Skipping {}: {}", path, e),
        }

        pb.inc(1);
        _processed += 1;
    }

    pb.finish_and_clear();
    if loss_count > 0 {
        println!(
            "Average training loss: {:.4}",
            total_loss / loss_count as f32
        );
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
    let processed_speakers = count_speakers(&train_files);
    println!("Processed {} speakers in this batch.", processed_speakers);
    println!("Number of speakers discovered: {}", net.output_size());
    for (i, files) in net.file_lists().iter().enumerate() {
        println!("Speaker {}: {} samples", i, files.len());
    }
}
