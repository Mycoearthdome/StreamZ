use hound;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, RwLock,
};
use streamz_rs::{

    average_vectors, batch_resample, compute_speaker_embeddings,
    extract_embedding_from_features, identify_speaker_cosine_feats,
    identify_speaker_with_threshold, load_and_resample_file, load_audio_samples,
    pretrain_from_features, set_wav_cache_enabled, train_from_files, wav_cache_enabled,
    FeatureExtractor, SimpleNeuralNet, DEFAULT_SAMPLE_RATE, FEATURE_SIZE, with_thread_extractor,

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
const TRAIN_EPOCHS: usize = 15;
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
    Some(cached_path.to_string_lossy().into_owned())
}

/// Convert all MP3 entries in the training list to cached WAV files.
fn precache_mp3_files(files: &mut [(String, Option<usize>)]) {
    for (path, _) in files.iter_mut() {
        if path.to_ascii_lowercase().ends_with(".mp3") {
            if let Some(new_path) = cache_mp3_as_wav(path) {
                *path = new_path;
            }
        }
    }
}

fn recompute_embeddings(
    net_arc: &Arc<RwLock<SimpleNeuralNet>>,
    extractor: &FeatureExtractor,
    speaker_features: &Arc<RwLock<HashMap<usize, Vec<Vec<f32>>>>>,
    speaker_embeddings: &Arc<RwLock<HashMap<usize, Vec<f32>>>>,
    embeddings: &Arc<Mutex<Vec<(Vec<f32>, f32, f32)>>>,
) {
    let net_snapshot = {
        let guard = net_arc.read().unwrap();
        guard.clone()
    };
    let feats_snapshot = {
        let guard = speaker_features.read().unwrap();
        guard.clone()
    };
    let mut new_embeds =
        compute_speaker_embeddings(&net_snapshot, extractor).unwrap_or_default();
    let mut avg_map: HashMap<usize, Vec<f32>> = HashMap::new();
    for (id, fs) in &feats_snapshot {
        avg_map.insert(*id, average_vectors(fs));
    }
    {
        let mut map = speaker_embeddings.write().unwrap();
        for (id, feat) in avg_map {
            map.insert(id, feat);
        }
        new_embeds = map
            .iter()
            .map(|(_, v)| (v.clone(), 0.0, 0.0))
            .collect();
    }
    {
        let mut guard = embeddings.lock().unwrap();
        *guard = new_embeds;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    std::thread::spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let deadlocks = parking_lot::deadlock::check_deadlock();
        if deadlocks.is_empty() {
            continue;
        }
        eprintln!("⚠️ Detected {} deadlocks!", deadlocks.len());
        for (i, threads) in deadlocks.iter().enumerate() {
            eprintln!("Deadlock #{}", i);
            for t in threads {
                eprintln!("Thread Id {:#?}", t.thread_id());
                eprintln!("{:#?}", t.backtrace());
            }
        }
    });
    let mut conf_threshold = DEFAULT_CONF_THRESHOLD;
    let mut eval_split = 0.2f32;
    let mut burn_in_limit: Option<usize> = None;
    let mut max_speakers: Option<usize> = None;
    let eval_mode = args.iter().any(|a| a == "--eval");
    let no_cache_wav = args.iter().any(|a| a == "--no-cache-wav");
    set_wav_cache_enabled(!no_cache_wav);
    let extractor = FeatureExtractor::new();
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

    // Convert all MP3 files to cached WAVs before proceeding when enabled
    if wav_cache_enabled() {
        precache_mp3_files(&mut train_files);
    }

    // Decode and resample all files in parallel once
    let path_list: Vec<String> = train_files.iter().map(|(p, _)| p.clone()).collect();
    let resampled_audio = batch_resample(&path_list);
    let feature_map: HashMap<String, Vec<Vec<f32>>> = resampled_audio
        .into_par_iter()
        .map(|(path, samples)| {
            let feats = extractor.extract(&samples);
            (path, feats)
        })
        .collect();

    let dataset_size = train_files.len();
    let burn_in_default = ((dataset_size as f32) * DEFAULT_BURN_IN_FRAC).ceil() as usize;
    let burn_in_limit_val = burn_in_limit.unwrap_or_else(|| burn_in_default.clamp(10, 50));
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
        let net_arc = Arc::new(RwLock::new(SimpleNeuralNet::new(
            FEATURE_SIZE,
            512,
            256,
            count_speakers(&train_files).max(1),
        )));
        if !train_refs.is_empty() {
            let out_sz = {
                let guard = net_arc.read().unwrap();
                guard.output_size()
            };
            let _ = train_from_files(
                net_arc.clone(),
                &train_refs,
                train_refs.len(),
                out_sz,
                TRAIN_EPOCHS,
                0.01,
                DROPOUT_PROB,
                BATCH_SIZE,
                &extractor,
            );
        }
        let mut net = match Arc::try_unwrap(net_arc) {
            Ok(m) => m.into_inner().unwrap(),
            Err(_) => panic!("Arc has other references"),
        };
        let total = eval_set.len();
        let mut correct = 0usize;
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fnc = 0usize;
        for (path, class) in &eval_set {
            match load_audio_samples(path) {
                Ok(samples) => {
                    match identify_speaker_with_threshold(
                        &net,
                        &samples,
                        conf_threshold,
                        &extractor,
                    ) {
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
        let net_arc = Arc::new(RwLock::new(SimpleNeuralNet::new(
            FEATURE_SIZE,
            512,
            256,
            num_speakers.max(1),
        )));
        let train_refs: Vec<(&str, usize)> = train_files
            .iter()
            .filter_map(|(p, c)| c.map(|cls| (p.as_str(), cls)))
            .collect();
        if !train_refs.is_empty() {
            let out_sz = {
                let guard = net_arc.read().unwrap();
                guard.output_size()
            };
            if let Err(e) = train_from_files(
                net_arc.clone(),
                &train_refs,
                train_files.len(),
                out_sz,
                TRAIN_EPOCHS,
                0.01,
                DROPOUT_PROB,
                BATCH_SIZE,
                &extractor,
            ) {
                eprintln!("Training failed: {}", e);
            }
        }
        match Arc::try_unwrap(net_arc) {
            Ok(m) => m.into_inner().unwrap(),
            Err(_) => panic!("Arc has other references"),
        }
    };

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
        *embed_vec = map
            .iter()
            .map(|(_, v)| (v.clone(), 0.0, 0.0))
            .collect();
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
    let embed_vec_snapshot = {
        let guard = embeddings.lock().unwrap();
        guard.clone()
    };

    train_files.par_iter_mut().for_each(|(path, class)| {
    with_thread_extractor(|extractor| {
        pb_arc.set_message(path.to_string());

        if let Some(windows) = feats_arc.get(path) {
            if windows.len() < 5 {
                eprintln!("Skipping {}, too short", path);
                pb_arc.inc(1);
                return;
            }
            let embeds = embed_vec_snapshot.clone();
            let count = loss_count.load(Ordering::SeqCst);
            let burn_phase = count < burn_in_limit_val;
            let dynamic_threshold = if burn_phase { 0.5 } else { conf_threshold };
            if burn_phase && class.is_none() {
                eprintln!("Burn-in: forcing new label for {}", path);
                let mut net = net_arc.write().unwrap();
                net.add_output_class();
                let new_label = net.output_size() - 1;
                *class = Some(new_label);
                net.record_training_file(new_label, path);
                let emb = {
                    let net_r = net_arc.read().unwrap();
                    extract_embedding_from_features(&net_r, windows)
                };
                speaker_features
                    .write()
                    .unwrap()
                    .entry(new_label)
                    .or_default()
                    .push(emb);
            }

            if let Some(label) = *class {
                // Known speaker: supervised training
                let mut net = net_arc.write().unwrap();
                let sz = net.output_size();
                let lr = if count < 1000 { 0.05 } else { 0.01 };
                let loss = pretrain_from_features(
                    &mut net,
                    windows,
                    label,
                    sz,
                    5,
                    lr,
                    DROPOUT_PROB,
                    BATCH_SIZE,
                );
                *total_loss.lock().unwrap() += loss;
                net.record_training_file(label, path);
                drop(net);
                let new_count = loss_count.fetch_add(1, Ordering::SeqCst) + 1;
                if new_count % 100 == 0 {
                    recompute_embeddings(
                        &net_arc,
                        extractor,
                        &speaker_features,
                        &speaker_embeddings,
                        &embeddings,
                    );
                }
                let emb = {
                    let net_r = net_arc.read().unwrap();
                    extract_embedding_from_features(&net_r, windows)
                };
                speaker_features
                    .write()
                    .unwrap()
                    .entry(label)
                    .or_default()
                    .push(emb);
            } else {
                // Unlabelled: try to match known speaker

                eprintln!("Embedding count: {}", embeds.len());
                let net_snapshot = {
                    let guard = net_arc.read().unwrap();
                    guard.clone()
                };
                if let Some(pred) =
                    identify_speaker_cosine_feats(&net_snapshot, &embeds, windows, dynamic_threshold)
                {
                    eprintln!("Path: {}, Predicted: {:?}, Threshold: {}", path, pred, dynamic_threshold);
                    *class = Some(pred);
                    let mut net = net_arc.write().unwrap();
                    let sz = net.output_size();
                    let lr = if count < 1000 { 0.05 } else { 0.01 };
                    let loss = pretrain_from_features(
                        &mut net,
                        windows,
                        pred,
                        sz,
                        5,
                        lr,
                        DROPOUT_PROB,
                        BATCH_SIZE,
                    );
                    *total_loss.lock().unwrap() += loss;
                    net.record_training_file(pred, path);
                    drop(net);
                    let new_count = loss_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if new_count % 100 == 0 {
                        recompute_embeddings(
                            &net_arc,
                            extractor,
                            &speaker_features,
                            &speaker_embeddings,
                            &embeddings,
                        );
                    }
                    let emb = {
                        let net_snapshot = {
                            let guard = net_arc.read().unwrap();
                            guard.clone()
                        };
                        extract_embedding_from_features(&net_snapshot, windows)
                    };
                    speaker_features
                        .write()
                        .unwrap()
                        .entry(pred)
                        .or_default()
                        .push(emb);
                } else {
                    eprintln!("Path: {}, no match above threshold {}", path, dynamic_threshold);
                    // New speaker: expand class
                    let mut net = net_arc.write().unwrap();
                    net.add_output_class();
                    let new_label = net.output_size() - 1;
                    *class = Some(new_label);
                    let sz = net.output_size();
                    let lr = if count < 1000 { 0.05 } else { 0.01 };
                    let loss = pretrain_from_features(
                        &mut net,
                        windows,
                        new_label,
                        sz,
                        5,
                        lr,
                        DROPOUT_PROB,
                        BATCH_SIZE,
                    );
                    *total_loss.lock().unwrap() += loss;
                    net.record_training_file(new_label, path);
                    drop(net);
                    let new_count = loss_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if new_count % 100 == 0 {
                        recompute_embeddings(
                            &net_arc,
                            extractor,
                            &speaker_features,
                            &speaker_embeddings,
                            &embeddings,
                        );
                    }
                    let emb = {
                        let net_snapshot = {
                            let guard = net_arc.read().unwrap();
                            guard.clone()
                        };
                        extract_embedding_from_features(&net_snapshot, windows)
                    };
                    speaker_features
                        .write()
                        .unwrap()
                        .entry(new_label)
                        .or_default()
                        .push(emb);
                }
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
    if count > 0 {
        println!("Average training loss: {:.4}", final_loss / count as f32);
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

    if let Err(e) = net.save(MODEL_PATH) {
        eprintln!("Failed to save model: {}", e);
    }
}
