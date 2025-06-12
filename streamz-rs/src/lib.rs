// use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
// use crossterm::event::{self, Event, KeyCode, KeyEventKind};
// use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use hound;
use mel_filter::{mel, NormalizationFactor};
use minimp3::{Decoder, Error as Mp3Error, Frame};
use ndarray::{s, Array1, Array2, Axis};
use ndarray_npy::{NpzReader, NpzWriter};
use rand::Rng;
use rayon::prelude::*;
use rustdct::DctPlanner;
use rustfft::{num_complex::Complex, FftPlanner};
// use rodio;
use rubato::{FftFixedInOut, Resampler};
use std::error::Error;
use std::fs::File;

const DEFAULT_SAMPLE_RATE: u32 = 44100;
pub const WINDOW_SIZE: usize = 1024;
const N_MELS: usize = 26;
pub const FEATURE_SIZE: usize = 13;
/// Default dropout probability applied during training.
pub const DEFAULT_DROPOUT: f32 = 0.2;

/// Apply simple data augmentation to raw i16 samples.
/// Adds small random gain and noise to each sample.
pub fn augment(samples: &[i16]) -> Vec<i16> {
    let mut rng = rand::thread_rng();
    samples
        .iter()
        .map(|&s| {
            let noise: f32 = rng.gen_range(-0.005..0.005);
            let gain: f32 = rng.gen_range(0.9..1.1);
            (s as f32 * gain + noise * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32)
                as i16
        })
        .collect()
}

/// Apply dropout to a slice of features in-place.
fn apply_dropout(features: &mut [f32], prob: f32) {
    if prob <= 0.0 {
        return;
    }
    let mut rng = rand::thread_rng();
    for v in features.iter_mut() {
        if rng.gen::<f32>() < prob {
            *v = 0.0;
        }
    }
}

/// Convert a raw i16 audio sample to a normalized f32 value in [-1.0, 1.0]
pub fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

/// Resample i16 samples to 44.1kHz using rubato
pub fn resample_to_44100(samples: &[i16], from_rate: u32) -> Result<Vec<i16>, Box<dyn Error>> {
    if from_rate == DEFAULT_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }
    let chunk_size = 1024;
    let input: Vec<Vec<f32>> = vec![samples.iter().map(|&s| i16_to_f32(s)).collect()];
    let mut resampler = FftFixedInOut::<f32>::new(
        from_rate as usize,
        DEFAULT_SAMPLE_RATE as usize,
        chunk_size,
        1,
    )?;
    let output = resampler.process(&input, None)?;
    let mut result = Vec::with_capacity(output[0].len());
    for frame in &output[0] {
        result.push((frame * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    Ok(result)
}

/// Split samples into windows and compute MFCC features for each window.
fn window_samples(samples: &[i16]) -> Vec<Vec<f32>> {
    let mel_filters = mel::<f32>(
        DEFAULT_SAMPLE_RATE as usize,
        WINDOW_SIZE,
        Some(N_MELS),
        None,
        None,
        false,
        NormalizationFactor::One,
    );
    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(WINDOW_SIZE);
    let mut dct_planner = DctPlanner::<f32>::new();
    let dct = dct_planner.plan_dct2(N_MELS);

    let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); WINDOW_SIZE];
    let mut features = Vec::new();

    for chunk in samples
        .chunks(WINDOW_SIZE)
        .filter(|c| c.len() == WINDOW_SIZE)
    {
        for (i, &val) in chunk.iter().enumerate() {
            buffer[i] = Complex::new(i16_to_f32(val), 0.0);
        }
        fft.process(&mut buffer);
        let mags: Vec<f32> = buffer
            .iter()
            .take(WINDOW_SIZE / 2 + 1)
            .map(|c| c.norm_sqr())
            .collect();

        let mut mel_energies = vec![0.0f32; N_MELS];
        for (i, filt) in mel_filters.iter().enumerate() {
            let mut sum = 0.0f32;
            for (j, &w) in filt.iter().enumerate() {
                sum += w * mags[j];
            }
            mel_energies[i] = (sum.max(1e-12)).ln();
        }

        let mut coeffs = mel_energies;
        dct.process_dct2(&mut coeffs);
        coeffs.truncate(FEATURE_SIZE);
        features.push(coeffs);
    }

    features
}

/// Pre-train the network with a slice of `i16` samples.
pub fn pretrain_network(
    net: &mut SimpleNeuralNet,
    samples: &[i16],
    target_class: usize,
    num_classes: usize,
    epochs: usize,
    lr: f32,
    dropout: f32,
) {
    let mut target = vec![0.0f32; num_classes];
    if target_class < num_classes {
        target[target_class] = 1.0;
    }
    let aug_samples = augment(samples);
    let windows = window_samples(&aug_samples);
    for _ in 0..epochs {
        for win in &windows {
            let mut feats = win.clone();
            apply_dropout(&mut feats, dropout);
            net.train(&feats, &target, lr);
        }
    }
}

/// Load all samples from a 16-bit mono WAV file.
pub fn load_wav_samples(path: &str) -> Result<Vec<i16>, Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    if reader.spec().bits_per_sample != 16 {
        return Err("Only 16-bit audio supported".into());
    }
    let sample_rate = reader.spec().sample_rate;
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    let samples = samples?;
    resample_to_44100(&samples, sample_rate)
}

/// Load samples from an MP3 file using the `minimp3` decoder. Returns the
/// decoded samples along with the detected sample rate.
pub fn load_mp3_samples(path: &str) -> Result<(Vec<i16>, u32), Box<dyn Error>> {
    let mut decoder = Decoder::new(File::open(path)?);
    let mut samples = Vec::new();
    let mut sample_rate = 0u32;
    loop {
        match decoder.next_frame() {
            Ok(Frame {
                data,
                sample_rate: sr,
                ..
            }) => {
                if sample_rate == 0 {
                    sample_rate = sr as u32;
                }
                samples.extend_from_slice(&data);
            }
            Err(Mp3Error::Eof) => break,
            Err(e) => return Err(Box::new(e)),
        }
    }
    if sample_rate == 0 {
        return Err("No frames decoded".into());
    }
    Ok((samples, sample_rate))
}

/// Load audio samples from either a WAV or MP3 file depending on the file
/// extension.
pub fn load_audio_samples(path: &str) -> Result<Vec<i16>, Box<dyn Error>> {
    if path.to_ascii_lowercase().ends_with(".mp3") {
        let (samples, sr) = load_mp3_samples(path)?;
        resample_to_44100(&samples, sr)
    } else {
        load_wav_samples(path)
    }
}

/// Read the sample rate and bit depth from an audio file.
/// Supports WAV and MP3 formats.
pub fn audio_metadata(path: &str) -> Result<(u32, u16), Box<dyn Error>> {
    if path.to_ascii_lowercase().ends_with(".mp3") {
        let mut decoder = Decoder::new(File::open(path)?);
        if let Ok(Frame { .. }) = decoder.next_frame() {
            Ok((DEFAULT_SAMPLE_RATE, 16))
        } else {
            Err("Unable to decode MP3".into())
        }
    } else {
        let spec = hound::WavReader::open(path)?.spec();
        Ok((DEFAULT_SAMPLE_RATE, spec.bits_per_sample))
    }
}

/// Train the network using a list of `(path, class)` tuples containing WAV files.
pub fn train_from_files(
    net: &mut SimpleNeuralNet,
    files: &[(&str, usize)],
    total_files: usize,
    num_speakers: usize,
    epochs: usize,
    lr: f32,
    dropout: f32,
) -> Result<(), Box<dyn Error>> {
    use indicatif::{ProgressBar, ProgressStyle};
    println!("Training on {} files individually", total_files);
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} ETA {eta}")
            .unwrap(),
    );

    for &(path, class) in files {
        pb.set_message(path.to_string());
        let (sample_rate, bits) = match audio_metadata(path) {
            Ok(meta) => meta,
            Err(e) => {
                eprintln!("Skipping {}: {}", path, e);
                continue;
            }
        };
        net.set_dataset_specs(sample_rate, bits);
        if bits != 16 {
            eprintln!("Skipping {}: Only 16-bit audio supported", path);
            continue;
        }
        for _ in 0..epochs {
            match load_audio_samples(path) {
                Ok(samples) => {
                    pretrain_network(net, &samples, class, num_speakers, 1, lr, dropout);
                    net.record_training_file(class, path);
                }
                Err(e) => {
                    eprintln!("Skipping {}: {}", path, e);
                    break;
                }
            }
        }
        pb.inc(1);
    }
    pb.finish_and_clear();
    Ok(())
}

/// Record and train the network on a list of prompt sentences.
/// Record each prompt sentence and train the network.
/// The full list is cycled `cycles` times before returning.

/// Simple feed-forward neural network operating on floating point vectors.
///
/// The original implementation used only a single hidden layer. To provide a
/// slightly "deeper" network without pulling in heavyweight dependencies we now
/// include a second hidden layer. This keeps the code self contained while
/// giving the model a bit more capacity.
pub struct SimpleNeuralNet {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    w3: Array2<f32>,
    b3: Array1<f32>,
    num_speakers: usize,
    /// List of training file paths for each speaker
    file_lists: Vec<Vec<String>>,
    sample_rate: u32,
    bits: u16,
}

impl SimpleNeuralNet {
    /// Create a new network with the given layer sizes. Two hidden layers are
    /// used to provide a slightly deeper model.
    pub fn new(input: usize, hidden1: usize, hidden2: usize, output: usize) -> Self {
        use rand::distributions::Uniform;
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-0.5, 0.5);
        let w2 = Array2::from_shape_fn((hidden1, hidden2), |_| rng.sample(dist));
        let b2 = Array1::zeros(hidden2);
        let w3 = Array2::from_shape_fn((hidden2, output), |_| rng.sample(dist));
        let b3 = Array1::zeros(output);
        Self {
            w1: Array2::from_shape_fn((input, hidden1), |_| rng.sample(dist)),
            b1: Array1::zeros(hidden1),
            w2,
            b2,
            w3,
            b3,
            num_speakers: output,
            file_lists: vec![Vec::new(); output],
            sample_rate: DEFAULT_SAMPLE_RATE,
            bits: 16,
        }
    }

    pub fn output_size(&self) -> usize {
        self.num_speakers
    }

    /// Add a new output class to the network by expanding the last layer
    pub fn add_output_class(&mut self) {
        use rand::distributions::Uniform;
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-0.5, 0.5);
        let hidden = self.w3.nrows();
        let mut new_w3 = Array2::<f32>::zeros((hidden, self.num_speakers + 1));
        for i in 0..self.num_speakers {
            let col = self.w3.column(i).to_owned();
            new_w3.column_mut(i).assign(&col);
        }
        for r in 0..hidden {
            new_w3[[r, self.num_speakers]] = rng.sample(dist);
        }
        self.w3 = new_w3;

        let mut new_b3 = Array1::<f32>::zeros(self.num_speakers + 1);
        for i in 0..self.num_speakers {
            new_b3[i] = self.b3[i];
        }
        self.b3 = new_b3;
        if self.file_lists.len() <= self.num_speakers {
            self.file_lists.push(Vec::new());
        }
        self.num_speakers += 1;
    }

    pub fn set_dataset_specs(&mut self, sample_rate: u32, bits: u16) {
        self.sample_rate = sample_rate;
        self.bits = bits;
    }

    /// Record a file path for the given speaker so it can be saved with the model
    pub fn record_training_file(&mut self, class: usize, path: &str) {
        if self.file_lists.len() <= class {
            self.file_lists.resize_with(class + 1, Vec::new);
        }
        if !self.file_lists[class].contains(&path.to_string()) {
            self.file_lists[class].push(path.to_string());
        }
    }

    /// Forward pass on a slice of f32 values
    pub fn forward(&self, bits: &[f32]) -> Vec<f32> {
        let x = Array1::from_vec(bits.to_vec());
        let h1 = (x.dot(&self.w1) + &self.b1).mapv(|v| v.tanh());
        let h2 = (h1.dot(&self.w2) + &self.b2).mapv(|v| v.tanh());
        let w3 = self.w3.slice(s![.., ..self.num_speakers]);
        let b3 = self.b3.slice(s![..self.num_speakers]);
        let out = h2.dot(&w3) + &b3;
        let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = out.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        (exp / sum).to_vec()
    }

    /// Extract the hidden layer activation as an embedding vector
    pub fn embed(&self, bits: &[f32]) -> Vec<f32> {
        let x = Array1::from_vec(bits.to_vec());
        let h1 = (x.dot(&self.w1) + &self.b1).mapv(|v| v.tanh());
        let h2 = (h1.dot(&self.w2) + &self.b2).mapv(|v| v.tanh());
        h2.to_vec()
    }

    /// Return the size of the embedding vector
    pub fn embedding_size(&self) -> usize {
        self.w2.ncols()
    }

    /// Single-step training using cross entropy loss
    pub fn train(&mut self, bits: &[f32], target: &[f32], lr: f32) {
        let x = Array1::from_vec(bits.to_vec());
        let t = Array1::from_vec(target.to_vec());
        let h1_pre = x.dot(&self.w1) + &self.b1;
        let h1 = h1_pre.mapv(|v| v.tanh());
        let h2_pre = h1.dot(&self.w2) + &self.b2;
        let h2 = h2_pre.mapv(|v| v.tanh());
        let w3 = self.w3.slice(s![.., ..self.num_speakers]);
        let b3 = self.b3.slice(s![..self.num_speakers]);
        let out_pre = h2.dot(&w3) + &b3;
        let max = out_pre.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = out_pre.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        let out = exp.mapv(|v| v / sum);

        let delta_out = &out - &t;
        let grad_w3 = h2
            .insert_axis(Axis(1))
            .dot(&delta_out.clone().insert_axis(Axis(0)));
        let grad_b3 = delta_out.clone();
        let delta_h2 = delta_out.dot(&w3.t()) * h2_pre.mapv(|v| 1.0 - v.tanh().powi(2));
        let grad_w2 = h1
            .insert_axis(Axis(1))
            .dot(&delta_h2.clone().insert_axis(Axis(0)));
        let grad_b2 = delta_h2.clone();
        let delta_h1 = delta_h2.dot(&self.w2.t()) * h1_pre.mapv(|v| 1.0 - v.tanh().powi(2));
        let grad_w1 = x
            .insert_axis(Axis(1))
            .dot(&delta_h1.clone().insert_axis(Axis(0)));
        let grad_b1 = delta_h1;

        {
            let mut w3_mut = self.w3.slice_mut(s![.., ..self.num_speakers]);
            w3_mut -= &(grad_w3 * lr);
        }
        {
            let mut b3_mut = self.b3.slice_mut(s![..self.num_speakers]);
            b3_mut -= &(grad_b3 * lr);
        }
        self.w2 -= &(grad_w2 * lr);
        self.b2 -= &(grad_b2 * lr);
        self.w1 -= &(grad_w1 * lr);
        self.b1 -= &(grad_b1 * lr);
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        let mut npz = NpzWriter::new(file);
        npz.add_array("w1", &self.w1)?;
        npz.add_array("b1", &self.b1)?;
        npz.add_array("w2", &self.w2)?;
        npz.add_array("b2", &self.b2)?;
        npz.add_array("sample_rate", &ndarray::arr1(&[self.sample_rate as i64]))?;
        npz.add_array("bits", &ndarray::arr1(&[self.bits as i64]))?;
        npz.add_array("num_speakers", &ndarray::arr1(&[self.num_speakers as i64]))?;
        for idx in 0..self.num_speakers {
            let w_name = format!("w3_{}", idx + 1);
            let b_name = format!("b3_{}", idx + 1);
            let w_col = self.w3.column(idx).to_owned();
            let b_val = Array1::<f32>::from_vec(vec![self.b3[idx]]);
            npz.add_array(&w_name, &w_col)?;
            npz.add_array(&b_name, &b_val)?;
        }
        for (idx, files) in self.file_lists.iter().take(self.num_speakers).enumerate() {
            let joined = files.join("\n");
            let arr = Array1::<u8>::from_vec(joined.as_bytes().to_vec());
            npz.add_array(&format!("speaker_{}_files", idx), &arr)?;
        }
        npz.finish()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut npz = NpzReader::new(file)?;
        let names = npz.names()?;
        let sample_rate: ndarray::Array1<i64> = npz.by_name("sample_rate")?;
        let bits: ndarray::Array1<i64> = npz.by_name("bits")?;
        let w1: Array2<f32> = npz.by_name("w1")?;
        let b1: Array1<f32> = npz.by_name("b1")?;
        let w2: Array2<f32> = npz.by_name("w2")?;
        let b2: Array1<f32> = npz.by_name("b2")?;
        let num_speakers_arr: Option<ndarray::Array1<i64>> =
            if names.iter().any(|n| n == "num_speakers.npy") {
                Some(npz.by_name("num_speakers")?)
            } else {
                None
            };

        let mut columns = Vec::new();
        let mut biases = Vec::new();
        let mut idx = 1;
        loop {
            let w_name = format!("w3_{}", idx);
            let b_name = format!("b3_{}", idx);
            let entry_w = format!("{}.npy", w_name);
            let entry_b = format!("{}.npy", b_name);
            if names.iter().any(|n| n == &entry_w) && names.iter().any(|n| n == &entry_b) {
                let w_col: Array1<f32> = npz.by_name(&w_name)?;
                let b_val: Array1<f32> = npz.by_name(&b_name)?;
                columns.push(w_col);
                biases.push(b_val[0]);
                idx += 1;
            } else {
                break;
            }
        }

        let mut num_outputs = columns.len();
        let hidden2 = w2.ncols();
        let mut w3 = Array2::<f32>::zeros((hidden2, num_outputs.max(1)));
        let mut b3 = Array1::<f32>::zeros(num_outputs.max(1));
        if !columns.is_empty() {
            for (i, col) in columns.into_iter().enumerate() {
                w3.column_mut(i).assign(&col);
            }
            for (i, val) in biases.into_iter().enumerate() {
                b3[i] = val;
            }
        } else if names.iter().any(|n| n == "w3.npy") {
            let w3_raw: Array2<f32> = npz.by_name("w3")?;
            let b3_raw: Array1<f32> = npz.by_name("b3")?;
            num_outputs = b3_raw.len();
            for i in 0..num_outputs {
                w3.column_mut(i).assign(&w3_raw.column(i));
                b3[i] = b3_raw[i];
            }
        }
        let outputs = if let Some(arr) = num_speakers_arr {
            arr[0] as usize
        } else if num_outputs > 0 {
            num_outputs
        } else {
            0
        };

        let mut file_lists = Vec::with_capacity(outputs);
        for i in 0..outputs {
            let name = format!("speaker_{}_files", i);
            let entry_name = format!("{}.npy", name);
            if names.iter().any(|n| n == &entry_name) {
                let arr: Array1<u8> = npz.by_name(&name)?;
                let text = String::from_utf8(arr.to_vec()).unwrap_or_default();
                let files = if text.is_empty() {
                    Vec::new()
                } else {
                    text.lines().map(|s| s.to_string()).collect()
                };
                file_lists.push(files);
            } else {
                file_lists.push(Vec::new());
            }
        }

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            num_speakers: outputs,
            file_lists,
            sample_rate: sample_rate[0] as u32,
            bits: bits[0] as u16,
        })
    }
}

/// Predict the speaker ID from a sample slice using the network
pub fn identify_speaker(net: &SimpleNeuralNet, sample: &[i16]) -> usize {
    let mut sums = vec![0.0f32; net.output_size()];
    for win in window_samples(sample) {
        let out = net.forward(&win);
        for (i, v) in out.iter().enumerate() {
            sums[i] += *v;
        }
    }
    sums.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Predict the speaker ID with a confidence threshold. If the maximum
/// probability across all windows is below `threshold`, `None` is returned.
pub fn identify_speaker_with_threshold(
    net: &SimpleNeuralNet,
    sample: &[i16],
    threshold: f32,
) -> Option<usize> {
    // If there's only a single speaker known, every prediction will trivially
    // have a confidence of 1.0. In that case we should treat the result as
    // "unknown" so new speakers can be added.
    if net.output_size() <= 1 {
        return None;
    }
    let mut sums = vec![0.0f32; net.output_size()];
    let mut count = 0f32;
    for win in window_samples(sample) {
        let out = net.forward(&win);
        for (i, v) in out.iter().enumerate() {
            sums[i] += *v;
        }
        count += 1.0;
    }
    if count == 0.0 {
        return None;
    }
    let (best_idx, best_val) = sums
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let confidence = best_val / count;
    if confidence >= threshold {
        Some(best_idx)
    } else {
        None
    }
}

/// Identify all speakers present in a sample using a per-window confidence
/// threshold. Each window is classified individually and the speaker with the
/// highest probability is counted if that probability exceeds `threshold`.
/// Speakers are returned in descending order of occurrences across windows.
pub fn identify_speaker_list(net: &SimpleNeuralNet, sample: &[i16], threshold: f32) -> Vec<usize> {
    let mut counts = vec![0usize; net.output_size()];
    for win in window_samples(sample) {
        let out = net.forward(&win);
        if let Some((best_idx, best_val)) = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if *best_val >= threshold {
                counts[best_idx] += 1;
            }
        }
    }
    let mut pairs: Vec<(usize, usize)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i, c))
        .collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs.into_iter().map(|(i, _)| i).collect()
}

/// Compute the average embedding vector for an audio sample
pub fn extract_embedding(net: &SimpleNeuralNet, sample: &[i16]) -> Vec<f32> {
    let mut sum = vec![0.0f32; net.embedding_size()];
    let mut count = 0f32;
    for win in window_samples(sample) {
        let emb = net.embed(&win);
        for (i, v) in emb.iter().enumerate() {
            sum[i] += *v;
        }
        count += 1.0;
    }
    if count > 0.0 {
        for v in &mut sum {
            *v /= count;
        }
    }
    sum
}

/// Calculate cosine similarity between two embedding vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Compute an average embedding vector for every known speaker.
/// Each speaker's embedding is the mean of embeddings for all recorded
/// training files belonging to that speaker. Missing or unreadable files
/// are skipped.
pub fn compute_speaker_embeddings(net: &SimpleNeuralNet) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let embeds: Vec<Vec<f32>> = net.file_lists[..net.output_size()]
        .par_iter()
        .map(|files| {
            let mut sum = vec![0.0f32; net.embedding_size()];
            let mut count = 0f32;
            for path in files {
                if let Ok(samples) = load_audio_samples(path) {
                    let emb = extract_embedding(net, &samples);
                    for (i, v) in emb.iter().enumerate() {
                        sum[i] += *v;
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in &mut sum {
                    *v /= count;
                }
            }
            sum
        })
        .collect();
    Ok(embeds)
}

/// Identify a speaker by comparing an embedding against known speaker
/// embeddings using cosine similarity. Returns `Some(index)` if the best
/// similarity exceeds `threshold`.
pub fn identify_speaker_cosine(
    net: &SimpleNeuralNet,
    speaker_embeds: &[Vec<f32>],
    sample: &[i16],
    threshold: f32,
) -> Option<usize> {
    if speaker_embeds.is_empty() {
        return None;
    }
    let emb = extract_embedding(net, sample);
    let (best_idx, best_val) = speaker_embeds
        .par_iter()
        .enumerate()
        .map(|(i, ref_vec)| (i, cosine_similarity(ref_vec, &emb)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    if best_val >= threshold {
        Some(best_idx)
    } else {
        None
    }
}
