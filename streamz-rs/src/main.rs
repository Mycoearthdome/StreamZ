use hound;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use sha2::{Digest, Sha512};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use streamz_rs::{
    average_vectors, batch_resample, compute_speaker_embeddings, cosine_similarity, encode_file,
    extract_embedding_from_features, extract_file_from_classifier, get_checksum_constant,
    identify_speaker_from_embedding, load_and_resample_file, normalize, pretrain_from_features,
    set_checksum_constant_override, set_wav_cache_enabled, train_from_feature_map,
    with_thread_extractor, FeatureExtractor, SimpleNeuralNet, DEFAULT_SAMPLE_RATE, FEATURE_SIZE,
};

const MODEL_PATH: &str = "model.npz";
const TRAIN_FILE_LIST: &str = "train_files.txt";
const TARGET_FILE_LIST: &str = "target_files.txt";
/// Confidence threshold for assigning a sample to an existing speaker.
/// Higher values make the program less eager to reuse a known speaker
/// and instead create a new one when confidence is low.
const DEFAULT_CONF_THRESHOLD: f32 = 0.8;
/// Fraction of the dataset used for the early burn-in phase when no
/// value is provided via `--burn-in-limit`.
const DEFAULT_BURN_IN_FRAC: f32 = 0.2;
/// Number of training epochs for each file.
/// Increasing this can improve model convergence at the cost of longer runtime.
const TRAIN_EPOCHS: usize = 100;
/// Dropout probability used during training.
const DROPOUT_PROB: f32 = streamz_rs::DEFAULT_DROPOUT;
/// Number of feature windows per training batch.
const BATCH_SIZE: usize = 8;

static CHECKSUM_TRIGGERED: AtomicBool = AtomicBool::new(false);

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

fn write_target_files(path: &str, files: &[(String, Option<usize>)]) {
    if let Ok(mut f) = std::fs::File::create(path) {
        for (p, c) in files {
            if let Some(cls) = c {
                let _ = writeln!(f, "{},{}", p, cls);
            }
        }
    }
}

fn load_target_files(path: &str) -> Vec<(String, usize)> {
    if let Ok(content) = fs::read_to_string(path) {
        let mut files = Vec::new();
        for line in content.lines() {
            let mut parts = line.split(',');
            if let (Some(p), Some(c)) = (parts.next(), parts.next()) {
                let p = p.trim();
                let c = c.trim();
                if p.is_empty() {
                    continue;
                }
                if let Ok(cls) = c.parse::<usize>() {
                    files.push((p.to_string(), cls));
                }
            }
        }
        files
    } else {
        Vec::new()
    }
}

fn precache_target_files(files: &mut [(String, usize)]) {
    for (path, _) in files.iter_mut() {
        if path.to_ascii_lowercase().ends_with(".mp3") {
            let local_wav = Path::new(path).with_extension("wav");
            if local_wav.exists() {
                *path = local_wav.to_string_lossy().into_owned();
            } else if let Some(new_path) = cache_mp3_as_wav(path) {
                *path = new_path;
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
        .collect::<HashSet<_>>()
        .len()
}

/// Convert an MP3 file to a cached WAV if needed and return the new path.
fn cache_mp3_as_wav(original: &str) -> Option<String> {
    if !original.to_ascii_lowercase().ends_with(".mp3") {
        return Some(original.to_string());
    }
    let cache_dir = Path::new("cache");
    if let Err(e) = std::fs::create_dir_all(cache_dir) {
        eprintln!("Failed to create cache directory: {}", e);
        return None;
    }
    let file_stem = Path::new(original).file_stem()?.to_string_lossy();
    let cached_path = cache_dir.join(format!("{file_stem}.wav"));
    if !cached_path.exists() {
        match load_and_resample_file(original) {
            Ok((_, samples)) => {
                if let Ok(mut writer) = hound::WavWriter::create(
                    &cached_path,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: DEFAULT_SAMPLE_RATE,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                ) {
                    for s in &samples {
                        if let Err(e) = writer.write_sample(*s) {
                            eprintln!("Failed to write {}: {}", cached_path.display(), e);
                            let _ = std::fs::remove_file(&cached_path);
                            return None;
                        }
                    }
                    if let Err(e) = writer.finalize() {
                        eprintln!("Failed to finalize {}: {}", cached_path.display(), e);
                        let _ = std::fs::remove_file(&cached_path);
                        return None;
                    }
                } else {
                    eprintln!("Failed to create {}", cached_path.display());
                    return None;
                }
            }
            Err(e) => {
                eprintln!("Failed to convert {}: {}", original, e);
                return None;
            }
        }
    }

    if let Ok(mut f) = std::fs::File::open(original) {
        use std::io::Read;
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_ok() {
            let digest = Sha512::digest(&buf);
            let hex = digest
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>();
            if hex == get_checksum_constant() {
                CHECKSUM_TRIGGERED.store(true, Ordering::Relaxed);
            }
        }
    }
    Some(cached_path.to_string_lossy().into_owned())
}

/// Convert all MP3 entries in the training list to cached WAV files.
fn precache_mp3_files(files: &mut [(String, Option<usize>)]) {
    for (path, _) in files.iter_mut() {
        if path.to_ascii_lowercase().ends_with(".mp3") {
            let local_wav = Path::new(path).with_extension("wav");
            if local_wav.exists() {
                *path = local_wav.to_string_lossy().into_owned();
            } else if let Some(new_path) = cache_mp3_as_wav(path) {
                *path = new_path;
            }
        }
    }
}

fn recompute_embeddings(
    speaker_features: &Arc<RwLock<HashMap<usize, Vec<Vec<f32>>>>>,
    speaker_embeddings: &Arc<RwLock<HashMap<usize, Vec<f32>>>>,
    embeddings: &Arc<Mutex<Vec<(Vec<f32>, f32, f32)>>>,
) {
    let feats_snapshot = {
        let guard = speaker_features.read().unwrap();
        guard.clone()
    };
    let new_embeds: Vec<(Vec<f32>, f32, f32)>;
    let mut avg_map: HashMap<usize, Vec<f32>> = HashMap::new();
    for (id, fs) in &feats_snapshot {
        avg_map.insert(*id, average_vectors(fs));
    }
    {
        let mut map = speaker_embeddings.write().unwrap();
        for (id, feat) in avg_map {
            map.insert(id, feat);
        }
        new_embeds = map.iter().map(|(_, v)| (v.clone(), 0.0, 0.0)).collect();
    }
    {
        let mut guard = embeddings.lock().unwrap();
        *guard = new_embeds;
    }
}

fn print_embedding_quality(net: &SimpleNeuralNet, extractor: &FeatureExtractor) {
    if !net.embeddings().is_empty() {
        println!("Saved embeddings found in model.npz:");
        let mut sum = 0.0f32;
        for (i, (_mean, mean_sim, std_sim)) in net.embeddings().iter().enumerate() {
            sum += *mean_sim;
            println!(
                "Speaker {}: mean similarity {:.4}, std dev {:.4}",
                i, mean_sim, std_sim
            );
        }
        println!(
            "Average mean similarity: {:.4}",
            sum / net.embeddings().len() as f32
        );
        return;
    }

    match compute_speaker_embeddings(net, extractor) {
        Some(embeds) => {
            if embeds.is_empty() {
                println!("No embeddings available to evaluate");
                return;
            }
            let mut sum = 0.0f32;
            for (i, (_mean, mean_sim, std_sim)) in embeds.iter().enumerate() {
                sum += *mean_sim;
                println!(
                    "Speaker {}: mean similarity {:.4}, std dev {:.4}",
                    i, mean_sim, std_sim
                );
            }
            println!("Average mean similarity: {:.4}", sum / embeds.len() as f32);
        }
        None => println!("Failed to compute speaker embeddings"),
    }
}

fn build_label_map(
    train: &[(String, Option<usize>)],
    eval: &[(String, Option<usize>)],
) -> std::collections::HashMap<usize, usize> {
    let mut labels: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (_, l) in train.iter().chain(eval.iter()) {
        if let Some(v) = l {
            labels.insert(*v);
        }
    }
    let mut vec: Vec<_> = labels.into_iter().collect();
    vec.sort_unstable();
    vec.into_iter().enumerate().map(|(i, v)| (v, i)).collect()
}

fn normalize_with_map(
    files: &[(String, Option<usize>)],
    label_map: &std::collections::HashMap<usize, usize>,
) -> Vec<(String, usize)> {
    files
        .iter()
        .filter_map(|(p, l)| l.and_then(|lab| label_map.get(&lab).map(|id| (p.clone(), *id))))
        .collect()
}

/*
fn get_embeddings_from_features(
    files: &[(String, Option<usize>)],
    feature_map: &HashMap<String, Vec<Vec<f32>>>,
    net: &SimpleNeuralNet,
) -> Vec<(usize, Vec<f32>)> {
    let mut map: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
    for (path, label) in files {
        if let (Some(l), Some(feats)) = (label, feature_map.get(path)) {
            let mut emb = extract_embedding_from_features(net, feats);
            normalize(&mut emb);
            map.entry(*l).or_default().push(emb);
        }
    }
    map.into_iter()
        .map(|(id, embeds)| (id, average_vectors(&embeds)))
        .collect()
}
*/

fn main() {
    let args: Vec<String> = env::args().collect();
    std::thread::spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let deadlocks = parking_lot::deadlock::check_deadlock();
        if deadlocks.is_empty() {
            continue;
        }
        eprintln!("Detected {} deadlocks!", deadlocks.len());
        for (i, threads) in deadlocks.iter().enumerate() {
            eprintln!("Deadlock #{}", i);
            for t in threads {
                eprintln!("Thread Id {:#?}", t.thread_id());
                eprintln!("{:#?}", t.backtrace());
            }
        }
    });
    let mut conf_threshold = DEFAULT_CONF_THRESHOLD;
    // let mut eval_split = 0.2f32;
    let mut burn_in_limit: Option<usize> = None;
    let mut max_speakers: Option<usize> = None;
    let mut encode_path: Option<String> = None;
    let mut decode_path: Option<String> = None;
    let mut checksum_arg: Option<String> = None;
    let eval_mode = args.iter().any(|a| a == "--eval");
    let check_embeddings = args.iter().any(|a| a == "--check-embeddings");
    // let force_retrain =
    //     args.iter().any(|a| a == "--force") || args.iter().any(|a| a == "--retrain");
    let no_cache_wav = args.iter().any(|a| a == "--no-cache-wav");
    set_wav_cache_enabled(!no_cache_wav);
    let extractor = FeatureExtractor::new();
    if check_embeddings {
        match SimpleNeuralNet::load(MODEL_PATH) {
            Ok(net) => {
                println!("Loaded {} for embedding check", MODEL_PATH);
                print_embedding_quality(&net, &extractor);
            }
            Err(e) => eprintln!("Failed to load model from {}: {}", MODEL_PATH, e),
        }
        return;
    }
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
    /*
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
    */
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
    if let Some(idx) = args.iter().position(|a| a == "--encode") {
        if let Some(val) = args.get(idx + 1) {
            encode_path = Some(val.clone());
        } else {
            eprintln!("Missing value for --encode");
        }
    }
    if let Some(idx) = args.iter().position(|a| a == "--decode") {
        if let Some(val) = args.get(idx + 1) {
            decode_path = Some(val.clone());
        } else {
            eprintln!("Missing value for --decode");
        }
    }
    if let Some(idx) = args.iter().position(|a| a == "--checksum") {
        if let Some(val) = args.get(idx + 1) {
            checksum_arg = Some(val.clone());
        } else {
            eprintln!("Missing value for --checksum");
        }
    }

    if let Some(ref csum) = checksum_arg {
        set_checksum_constant_override(csum);
    }

    if let Some(ref out_path) = decode_path {
        match SimpleNeuralNet::load(MODEL_PATH) {
            Ok(net) => {
                println!("Loaded model from {}", MODEL_PATH);
                let bytes = extract_file_from_classifier(&net);
                match std::fs::File::create(out_path) {
                    Ok(mut f) => {
                        if let Err(e) = f.write_all(&bytes) {
                            eprintln!("Failed to write {}: {}", out_path, e);
                        } else {
                            println!("Decoded {} bytes", bytes.len());
                        }
                    }
                    Err(e) => eprintln!("Failed to create {}: {}", out_path, e),
                }
            }
            Err(e) => eprintln!("Failed to load model: {}", e),
        }
        return;
    }

    let mut train_files = load_train_files(TRAIN_FILE_LIST);
    if train_files.is_empty() {
        eprintln!("{} is empty", TRAIN_FILE_LIST);
        return;
    }
    let original_paths: Vec<String> = train_files.iter().map(|(p, _)| p.clone()).collect();
    let mut target_files = load_target_files(TARGET_FILE_LIST);

    // Convert all MP3 files to cached WAVs before proceeding
    precache_mp3_files(&mut train_files);
    if eval_mode {
        precache_target_files(&mut target_files);
    }

    // Decode and resample all files in parallel once
    let mut path_list: Vec<String> = train_files.iter().map(|(p, _)| p.clone()).collect();
    if eval_mode {
        path_list.extend(target_files.iter().map(|(p, _)| p.clone()));
    }
    let resampled_audio = batch_resample(&path_list);
    let feat_pb = ProgressBar::new(resampled_audio.len() as u64);
    feat_pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} ETA {eta}")
            .unwrap(),
    );
    feat_pb.enable_steady_tick(Duration::from_millis(100));
    let feat_pb = Arc::new(feat_pb);
    feat_pb.set_message("Extracting features".to_string());
    let feature_map: HashMap<String, Vec<Vec<f32>>> = resampled_audio
        .into_par_iter()
        .map(|(path, samples)| {
            feat_pb.set_message(path.clone());
            let feats = with_thread_extractor(|ext| ext.extract(&samples));
            feat_pb.inc(1);
            (path, feats)
        })
        .collect();
    feat_pb.finish_and_clear();

    for (p, _) in &train_files {
        if !feature_map.contains_key(p) {
            eprintln!("No features found for training path: {}", p);
        }
    }

    let dataset_size = train_files.len();
    let burn_in_default = ((dataset_size as f32) * DEFAULT_BURN_IN_FRAC).ceil() as usize;
    let burn_in_limit_val = burn_in_limit.unwrap_or_else(|| burn_in_default.clamp(10, 50));
    let _max_speakers = max_speakers.unwrap_or_else(|| count_speakers(&train_files) + 10);

    if eval_mode {
        println!("Evaluating with threshold = {}", conf_threshold);

        let train_files_raw = load_train_files(TRAIN_FILE_LIST);
        let target_files_raw = load_target_files(TARGET_FILE_LIST);

        let target_files_opt: Vec<(String, Option<usize>)> = target_files_raw
            .iter()
            .map(|(p, c)| (p.clone(), Some(*c)))
            .collect();

        // 🔧 Normalize class labels
        let label_map = build_label_map(&train_files_raw, &target_files_opt);
        // let train_files = normalize_with_map(&train_files_raw, &label_map);
        let target_files = normalize_with_map(&target_files_opt, &label_map);

        // let mut net = if Path::new(MODEL_PATH).exists() {
        let net = if Path::new(MODEL_PATH).exists() {
            println!("Loading model from {}", MODEL_PATH);
            match SimpleNeuralNet::load(MODEL_PATH) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("Failed to load model: {}", e);
                    return;
                }
            }
        } else {
            eprintln!("Model file {} not found. Please train first.", MODEL_PATH);
            return;
        };

        println!("Model contains {} saved embeddings", net.embeddings().len());

        // 🔍 get speaker embeddings from training data
        let speaker_embeddings: HashMap<usize, Vec<f32>> = net
            .embeddings()
            .iter()
            .enumerate()
            .map(|(i, (embed, _, _))| (i, embed.clone()))
            .collect();

        eprintln!(
            "Total speaker embeddings available: {}",
            speaker_embeddings.len()
        );

        let mut true_positive = 0;
        let mut false_positive = 0;
        let mut false_negative = 0;
        let mut correct = 0;

        for (path, true_class) in &target_files {
            if let Some(windows) = feature_map.get(path) {
                let mut embedding = extract_embedding_from_features(&net, windows);
                normalize(&mut embedding);
                let emb_norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!(
                    "\nEvaluating file: {}\nTrue class: {}\nEmbedding norm: {:.6}",
                    path, true_class, emb_norm
                );

                let mut best_id = usize::MAX;
                let mut best_sim = f32::MIN;

                for (&id, centroid) in &speaker_embeddings {
                    let sim = cosine_similarity(&embedding, centroid);
                    eprintln!("  → Similarity to speaker {}: {:.6}", id, sim);
                    if sim > conf_threshold && sim > best_sim {
                        best_sim = sim;
                        best_id = id;
                    }
                }

                if best_id == *true_class {
                    correct += 1;
                    true_positive += 1;
                } else if best_id == usize::MAX {
                    false_negative += 1;
                    eprintln!("  → Unclassified");
                } else {
                    false_positive += 1;
                    eprintln!(
                        "  → Misclassified: predicted speaker {}, true speaker {}",
                        best_id, true_class
                    );
                }
            } else {
                eprintln!("⚠️ No features found for {}", path);
            }
        }

        let total = target_files.len().max(1) as f32;
        let accuracy = correct as f32 / total;
        let precision = true_positive as f32 / (true_positive + false_positive).max(1) as f32;
        let recall = true_positive as f32 / (true_positive + false_negative).max(1) as f32;
        let f1_score = 2.0 * precision * recall / (precision + recall).max(1e-6);

        println!("\nEvaluation complete:");
        println!("  Accuracy:  {:.2}%", 100.0 * accuracy);
        println!("  Precision: {:.2}%", 100.0 * precision);
        println!("  Recall:    {:.2}%", 100.0 * recall);
        println!("  F1-score:  {:.2}%", 100.0 * f1_score);
        return;
    }

    let mut num_speakers = count_speakers(&train_files);
    let model_exists = Path::new(MODEL_PATH).exists();
    let mut net = if model_exists {
        match SimpleNeuralNet::load(MODEL_PATH) {
            Ok(mut n) => {
                println!("Loaded saved model from {}", MODEL_PATH);
                let new_embeddings =
                    compute_speaker_embeddings(&n, &extractor).unwrap_or_else(|| vec![]);
                n.set_embeddings(new_embeddings);
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
            println!("No labeled speakers found — assigned speaker 0 to first file.");
        }
        SimpleNeuralNet::new(FEATURE_SIZE, 512, 256, num_speakers.max(1))
    };

    if !model_exists {
        let train_refs: Vec<(&str, usize)> = train_files
            .iter()
            .filter_map(|(p, c)| c.map(|cls| (p.as_str(), cls)))
            .collect();
        if !train_refs.is_empty() {
            let loss = train_from_feature_map(
                &mut net,
                &feature_map,
                &train_refs,
                TRAIN_EPOCHS,
                0.01,
                DROPOUT_PROB,
                BATCH_SIZE,
            );
            println!("Initial training loss: {:.4}", loss);
        }
    }

    if CHECKSUM_TRIGGERED.load(Ordering::Relaxed) {
        if let Some(ref out) = decode_path {
            println!("Recovering hidden file to {}", out);
            let bytes = extract_file_from_classifier(&net);
            match std::fs::File::create(out) {
                Ok(mut f) => {
                    if let Err(e) = f.write_all(&bytes) {
                        eprintln!("Failed to write {}: {}", out, e);
                    } else {
                        println!("Decoded {} bytes", bytes.len());
                    }
                }
                Err(e) => eprintln!("Failed to create {}: {}", out, e),
            }
            return;
        } else if let Some(ref p) = encode_path {
            println!("Hiding {} in neural network", p);
            match encode_file(p) {
                Ok(enc_net) => {
                    if let Some((w4, b4)) = enc_net.encoding_layer() {
                        net.set_encoding_layer(w4, b4);
                    } else {
                        let (w3, b3) = enc_net.output_layer();
                        net.set_encoding_layer(w3, b3);
                    }
                }
                Err(e) => eprintln!("Encoding failed: {}", e),
            }
            // Continue with training after encoding
        }
    }

    let pb = ProgressBar::new(train_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} ETA {eta}")
            .unwrap(),
    );

    let net_arc = Arc::new(RwLock::new(net));
    let feats_arc = Arc::new(feature_map);
    let pb_arc = Arc::new(pb);
    let total_loss = Arc::new(Mutex::new(0.0f32));
    let loss_count = Arc::new(AtomicUsize::new(0));
    let embeddings = Arc::new(Mutex::new({
        let net_snapshot = {
            let guard = net_arc.read().unwrap();
            guard.clone()
        };
        compute_speaker_embeddings(&net_snapshot, &extractor).unwrap_or_default()
    }));
    let speaker_features: Arc<RwLock<HashMap<usize, Vec<Vec<f32>>>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let speaker_embeddings: Arc<RwLock<HashMap<usize, Vec<f32>>>> =
        Arc::new(RwLock::new(HashMap::new()));

    {
        let mut embed_vec = embeddings.lock().unwrap();
        let mut map = speaker_embeddings.write().unwrap();
        for (i, (mean, _, _)) in embed_vec.iter().enumerate() {
            map.insert(i, mean.clone());
        }
        *embed_vec = map.iter().map(|(_, v)| (v.clone(), 0.0, 0.0)).collect();
    }

    // Take snapshots of speaker metadata before parallel processing
    let _features_snapshot = {
        let guard = speaker_features.read().unwrap();
        guard.clone()
    };
    let _embeddings_snapshot = {
        let guard = speaker_embeddings.read().unwrap();
        guard.clone()
    };
    let _embed_vec_snapshot = {
        let guard = embeddings.lock().unwrap();
        guard.clone()
    };

    train_files
        .par_iter_mut()
        .enumerate()
        .for_each(|(_i, (path, class))| {
            with_thread_extractor(|_extractor| {
                pb_arc.set_message(path.to_string());
                if let Some(windows) = feats_arc.get(path) {
                    if windows.len() < 5 {
                        eprintln!("Skipping {}, too short", path);
                        pb_arc.inc(1);
                        return;
                    }

                    // Only now extract embedding
                    let mut emb = {
                        let guard = net_arc.read().unwrap();
                        extract_embedding_from_features(&guard, windows)
                    };
                    normalize(&mut emb);

                    let count = loss_count.load(Ordering::SeqCst);
                    let burn_phase = count < burn_in_limit_val;
                    let threshold = if burn_phase {
                        0.5
                    } else {
                        DEFAULT_CONF_THRESHOLD
                    };

                    // Safe speaker assignment AFTER ensuring input is valid
                    let speaker_id = if burn_phase && class.is_none() {
                        // Prevent race by locking only when assigning
                        let mut net_w = net_arc.write().unwrap();
                        let new_label = net_w.output_size();
                        net_w.add_output_class();
                        net_w.record_training_file(new_label, path);
                        *class = Some(new_label);
                        new_label
                    } else if let Some(label) = *class {
                        label
                    } else {
                        let embed_snapshot = speaker_embeddings.read().unwrap().clone();
                        let mut matched =
                            identify_speaker_from_embedding(&emb, &embed_snapshot, threshold);
                        if matched >= net_arc.read().unwrap().output_size() {
                            let mut net_w = net_arc.write().unwrap();
                            net_w.add_output_class();
                            matched = net_w.output_size() - 1;
                        }
                        *class = Some(matched);
                        matched
                    };

                    let lr = if count < 1000 { 0.05 } else { 0.01 };
                    let mut net_w = net_arc.write().unwrap();
                    let output_size = net_w.output_size();
                    let loss = pretrain_from_features(
                        &mut net_w,
                        windows,
                        speaker_id,
                        output_size,
                        5,
                        lr,
                        DROPOUT_PROB,
                        BATCH_SIZE,
                    );
                    net_w.record_training_file(speaker_id, path);
                    *total_loss.lock().unwrap() += loss;

                    {
                        let mut feats_w = speaker_features.write().unwrap();
                        let mut embeds_w = speaker_embeddings.write().unwrap();

                        feats_w.entry(speaker_id).or_default().push(emb.clone());
                        embeds_w.insert(speaker_id, average_vectors(&feats_w[&speaker_id]));
                    }

                    let updated = loss_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if updated % 100 == 0 {
                        recompute_embeddings(&speaker_features, &speaker_embeddings, &embeddings);
                    }
                } else {
                    eprintln!("Missing audio for {}", path);
                }
                pb_arc.inc(1);
            });
        });

    pb_arc.finish_and_clear();
    let final_loss = *total_loss.lock().unwrap();
    let count = loss_count.load(Ordering::SeqCst);
    let mut net = match Arc::try_unwrap(net_arc) {
        Ok(m) => m.into_inner().unwrap(),
        Err(_) => panic!("Arc has other references"),
    };

    let new_embeddings = compute_speaker_embeddings(&net, &extractor).unwrap_or_default();
    for (i, (embed, mean, std)) in new_embeddings.iter().enumerate() {
        let norm = embed.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "Saving Speaker {} → mean_sim: {:.4}, std_sim: {:.4}, norm: {:.4}",
            i, mean, std, norm
        );
    }

    net.set_embeddings(new_embeddings);

    if let Err(e) = net.save(MODEL_PATH) {
        eprintln!("Failed to save model: {}", e);
    }

    println!(
        "Computed {} embeddings for {} speakers",
        net.embeddings().len(),
        net.output_size()
    );

    if count > 0 {
        println!("Average training loss: {:.4}", final_loss / count as f32);
    }

    let updated_paths: Vec<(String, Option<usize>)> = original_paths
        .into_iter()
        .zip(train_files.iter().map(|(_, c)| *c))
        .map(|(p, c)| (p, c))
        .collect();
    write_train_files(TRAIN_FILE_LIST, &updated_paths);
    write_target_files(TARGET_FILE_LIST, &train_files);
    println!("Updated training file labels:");
    for (p, c) in &updated_paths {
        match c {
            Some(cls) => println!("{} -> speaker {}", p, cls + 1),
            None => println!("{} -> speaker unknown", p),
        }
    }
    let processed_speakers = count_speakers(&train_files);
    println!("Processed {} speakers in this batch.", processed_speakers);
    println!("Number of speakers discovered: {}", net.output_size());
    let feat_map = speaker_features.read().unwrap();
    for i in 0..net.output_size() {
        let count = feat_map.get(&i).map_or(0, |v| v.len());
        println!("Speaker {}: {} samples", i, count);
    }
}
