// use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
// use crossterm::event::{self, Event, KeyCode, KeyEventKind};
// use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use hound;
use minimp3::{Decoder, Error as Mp3Error, Frame};
use ndarray::{s, Array1, Array2, Axis};
use ndarray_npy::{NpzReader, NpzWriter};
use rand::Rng;
// use rodio;
use std::error::Error;
use std::fs::File;

const DEFAULT_SAMPLE_RATE: u32 = 44100;
pub const WINDOW_SIZE: usize = 256;
pub const MAX_SPEAKERS: usize = 153;

/// Convert a raw i16 audio sample to a normalized f32 value in [-1.0, 1.0]
pub fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

/// Split samples into consecutive windows of `WINDOW_SIZE` normalized floats
fn window_samples(samples: &[i16]) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    samples
        .chunks(WINDOW_SIZE)
        .filter(|c| c.len() == WINDOW_SIZE)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|c| {
            let floats: Vec<f32> = c.iter().copied().map(i16_to_f32).collect();
            let mean = floats.iter().copied().sum::<f32>() / floats.len() as f32;
            let var = floats
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                / floats.len() as f32;
            let std = var.sqrt().max(1e-6);
            floats.iter().map(|&v| (v - mean) / std).collect()
        })
        .collect()
}

/// Pre-train the network with a slice of `i16` samples.
pub fn pretrain_network(
    net: &mut SimpleNeuralNet,
    samples: &[i16],
    target_class: usize,
    num_classes: usize,
    epochs: usize,
    lr: f32,
) {
    let mut target = vec![0.0f32; num_classes];
    if target_class < num_classes {
        target[target_class] = 1.0;
    }
    let windows = window_samples(samples);
    for _ in 0..epochs {
        for win in &windows {
            net.train(win, &target, lr);
        }
    }
}

/// Load all samples from a 16-bit mono WAV file.
pub fn load_wav_samples(path: &str) -> Result<Vec<i16>, Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    if reader.spec().bits_per_sample != 16 {
        return Err("Only 16-bit audio supported".into());
    }
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    Ok(samples?)
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
        let (samples, _) = load_mp3_samples(path)?;
        Ok(samples)
    } else {
        load_wav_samples(path)
    }
}

/// Read the sample rate and bit depth from an audio file.
/// Supports WAV and MP3 formats.
pub fn audio_metadata(path: &str) -> Result<(u32, u16), Box<dyn Error>> {
    if path.to_ascii_lowercase().ends_with(".mp3") {
        let mut decoder = Decoder::new(File::open(path)?);
        if let Ok(Frame { sample_rate, .. }) = decoder.next_frame() {
            Ok((sample_rate as u32, 16))
        } else {
            Err("Unable to decode MP3".into())
        }
    } else {
        let spec = hound::WavReader::open(path)?.spec();
        Ok((spec.sample_rate, spec.bits_per_sample))
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
                    pretrain_network(net, &samples, class, num_speakers, 1, lr);
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

/// Simple feed-forward neural network operating on floating point vectors
pub struct SimpleNeuralNet {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    num_speakers: usize,
    /// List of training file paths for each speaker
    file_lists: Vec<Vec<String>>,
    sample_rate: u32,
    bits: u16,
}

impl SimpleNeuralNet {
    /// Create a new network with the given layer sizes
    pub fn new(input: usize, hidden: usize, output: usize) -> Self {
        use rand::distributions::Uniform;
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-0.5, 0.5);
        let w2 = Array2::from_shape_fn((hidden, MAX_SPEAKERS), |_| rng.sample(dist));
        let b2 = Array1::zeros(MAX_SPEAKERS);
        Self {
            w1: Array2::from_shape_fn((input, hidden), |_| rng.sample(dist)),
            b1: Array1::zeros(hidden),
            w2,
            b2,
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
        if self.num_speakers >= MAX_SPEAKERS {
            return;
        }
        use rand::distributions::Uniform;
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-0.5, 0.5);
        let hidden = self.w2.nrows();
        for r in 0..hidden {
            self.w2[[r, self.num_speakers]] = rng.sample(dist);
        }
        self.b2[self.num_speakers] = 0.0;
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
        let h = (x.dot(&self.w1) + &self.b1).mapv(|v| v.tanh());
        let w2 = self.w2.slice(s![.., ..self.num_speakers]);
        let b2 = self.b2.slice(s![..self.num_speakers]);
        let out = h.dot(&w2) + &b2;
        let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = out.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        (exp / sum).to_vec()
    }

    /// Single-step training using cross entropy loss
    pub fn train(&mut self, bits: &[f32], target: &[f32], lr: f32) {
        let x = Array1::from_vec(bits.to_vec());
        let t = Array1::from_vec(target.to_vec());
        let h_pre = x.dot(&self.w1) + &self.b1;
        let h = h_pre.mapv(|v| v.tanh());
        let w2 = self.w2.slice(s![.., ..self.num_speakers]);
        let b2 = self.b2.slice(s![..self.num_speakers]);
        let out_pre = h.dot(&w2) + &b2;
        let max = out_pre.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = out_pre.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        let out = exp.mapv(|v| v / sum);

        let delta_out = &out - &t;
        let grad_w2 = h
            .insert_axis(Axis(1))
            .dot(&delta_out.clone().insert_axis(Axis(0)));
        let grad_b2 = delta_out.clone();
        let delta_h = delta_out.dot(&w2.t()) * h_pre.mapv(|v| 1.0 - v.tanh().powi(2));
        let grad_w1 = x
            .insert_axis(Axis(1))
            .dot(&delta_h.clone().insert_axis(Axis(0)));
        let grad_b1 = delta_h;

        {
            let mut w2_mut = self.w2.slice_mut(s![.., ..self.num_speakers]);
            w2_mut -= &(grad_w2 * lr);
        }
        {
            let mut b2_mut = self.b2.slice_mut(s![..self.num_speakers]);
            b2_mut -= &(grad_b2 * lr);
        }
        self.w1 -= &(grad_w1 * lr);
        self.b1 -= &(grad_b1 * lr);
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        let mut npz = NpzWriter::new(file);
        npz.add_array("w1", &self.w1)?;
        npz.add_array("b1", &self.b1)?;
        npz.add_array("sample_rate", &ndarray::arr1(&[self.sample_rate as i64]))?;
        npz.add_array("bits", &ndarray::arr1(&[self.bits as i64]))?;
        npz.add_array("num_speakers", &ndarray::arr1(&[self.num_speakers as i64]))?;
        for idx in 0..self.num_speakers {
            let w_name = format!("w{}", idx + 1);
            let b_name = format!("b{}", idx + 1);
            let w_col = self.w2.column(idx).to_owned();
            let b_val = Array1::<f32>::from_vec(vec![self.b2[idx]]);
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
            let w_name = format!("w{}", idx);
            let b_name = format!("b{}", idx);
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
        let hidden = w1.ncols();
        let mut w2 = Array2::<f32>::zeros((hidden, MAX_SPEAKERS));
        let mut b2 = Array1::<f32>::zeros(MAX_SPEAKERS);
        if !columns.is_empty() {
            for (i, col) in columns.into_iter().enumerate() {
                w2.column_mut(i).assign(&col);
            }
            for (i, val) in biases.into_iter().enumerate() {
                b2[i] = val;
            }
        } else if names.iter().any(|n| n == "w2.npy") {
            let w2_raw: Array2<f32> = npz.by_name("w2")?;
            let b2_raw: Array1<f32> = npz.by_name("b2")?;
            num_outputs = b2_raw.len();
            for i in 0..num_outputs {
                w2.column_mut(i).assign(&w2_raw.column(i));
                b2[i] = b2_raw[i];
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
