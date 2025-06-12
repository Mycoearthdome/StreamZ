// use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
// use crossterm::event::{self, Event, KeyCode, KeyEventKind};
// use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use hound;
use ndarray::{Array1, Array2, Axis};
use ndarray_npy::{NpzReader, NpzWriter};
use rand::Rng;
// use rodio;
use std::error::Error;
use std::fs::File;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};

const DEFAULT_SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 1;
pub const WINDOW_SIZE: usize = 256;

/// Convert a 16-bit sample to a vector of bits represented as f32 values (0.0 or 1.0)
pub fn i16_to_bits(val: i16) -> [f32; 16] {
    let mut bits = [0.0f32; 16];
    let val_u16 = val as u16;
    for i in 0..16 {
        bits[i] = if (val_u16 >> i) & 1 == 1 { 1.0 } else { 0.0 };
    }
    bits
}

/// Convert a vector of bits back into a 16-bit sample
pub fn bits_to_i16(bits: &[f32]) -> i16 {
    let mut value: u16 = 0;
    for i in 0..16 {
        if bits[i] > 0.5 {
            value |= 1 << i;
        }
    }
    value as i16
}

/// Convert a raw i16 audio sample to a normalized f32 value in [-1.0, 1.0]
pub fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

/// Convert a normalized f32 value back into an i16 audio sample
pub fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// Split samples into consecutive windows of `WINDOW_SIZE` normalized floats
fn window_samples(samples: &[i16]) -> Vec<Vec<f32>> {
    samples
        .chunks(WINDOW_SIZE)
        .filter(|c| c.len() == WINDOW_SIZE)
        .map(|c| c.iter().map(|&s| i16_to_f32(s)).collect())
        .collect()
}

/// Remove the estimated ambient noise level from a raw sample.
fn subtract_baseline(sample: i16, baseline: f32) -> f32 {
    let sign = if sample >= 0 { 1.0 } else { -1.0 };
    let mut val = sample.abs() as f32 - baseline;
    if val < 0.0 {
        val = 0.0;
    }
    val * sign
}

/// Apply a simple low-pass filter to reduce high frequency noise.
fn low_pass_filter(sample: f32, prev: &mut f32, alpha: f32) -> f32 {
    let filtered = alpha * sample + (1.0 - alpha) * *prev;
    *prev = filtered;
    filtered
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
        return Err("Only 16-bit WAV files supported".into());
    }
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    Ok(samples?)
}

/// Determine the sample rate and bit depth of the provided WAV files.
/// Returns an error if the files have mismatched specs.
fn detect_dataset_specs(files: &[(&str, usize)]) -> Result<(u32, u16), Box<dyn Error>> {
    let mut iter = files.iter();
    let first = iter.next().ok_or("No training files provided")?;
    let spec = hound::WavReader::open(first.0)?.spec();
    for (path, _) in iter {
        let this = hound::WavReader::open(path)?.spec();
        if this.sample_rate != spec.sample_rate || this.bits_per_sample != spec.bits_per_sample {
            return Err(format!("Inconsistent WAV specs between {} and {}", first.0, path).into());
        }
    }
    Ok((spec.sample_rate, spec.bits_per_sample))
}

/// Train the network using a list of `(path, class)` tuples containing WAV files.
pub fn train_from_files(
    net: &mut SimpleNeuralNet,
    files: &[(&str, usize)],
    num_speakers: usize,
    epochs: usize,
    lr: f32,
) -> Result<(), Box<dyn Error>> {
    let (sample_rate, bits) = detect_dataset_specs(files)?;
    println!(
        "Detected training data: {} Hz, {} bits per sample",
        sample_rate, bits
    );
    net.set_dataset_specs(sample_rate, bits);
    if bits != 16 {
        return Err("Only 16-bit WAV files supported".into());
    }
    for _ in 0..epochs {
        for &(path, class) in files {
            let samples = load_wav_samples(path)?;
            pretrain_network(net, &samples, class, num_speakers, 1, lr);
        }
    }
    Ok(())
}


/// Record and train the network on a list of prompt sentences.
/// Record each prompt sentence and train the network.
/// The full list is cycled `cycles` times before returning.

/// Asynchronous generator simulating a MIMO bit stream
pub struct MIMOStream {
    num_bits: usize,
    delay: Duration,
}

impl MIMOStream {
    pub fn new(num_bits: usize, delay_ms: u64) -> Self {
        Self {
            num_bits,
            delay: Duration::from_millis(delay_ms),
        }
    }

    /// Return a vector of random bits after a small delay
    pub async fn get_input_bits(&self) -> Vec<f32> {
        sleep(self.delay).await;
        let mut rng = rand::thread_rng();
        (0..self.num_bits)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect()
    }
}

/// Simple feed-forward neural network operating on floating point vectors
pub struct SimpleNeuralNet {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    sample_rate: u32,
    bits: u16,
}

impl SimpleNeuralNet {
    /// Create a new network with the given layer sizes
    pub fn new(input: usize, hidden: usize, output: usize) -> Self {
        use rand::distributions::Uniform;
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-0.5, 0.5);
        Self {
            w1: Array2::from_shape_fn((input, hidden), |_| rng.sample(dist)),
            b1: Array1::zeros(hidden),
            w2: Array2::from_shape_fn((hidden, output), |_| rng.sample(dist)),
            b2: Array1::zeros(output),
            sample_rate: DEFAULT_SAMPLE_RATE,
            bits: 16,
        }
    }

    pub fn output_size(&self) -> usize {
        self.b2.len()
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn bits(&self) -> u16 {
        self.bits
    }

    pub fn set_dataset_specs(&mut self, sample_rate: u32, bits: u16) {
        self.sample_rate = sample_rate;
        self.bits = bits;
    }

    /// Forward pass on a slice of f32 values
    pub fn forward(&self, bits: &[f32]) -> Vec<f32> {
        let x = Array1::from_vec(bits.to_vec());
        let h = (x.dot(&self.w1) + &self.b1).mapv(|v| v.tanh());
        let out = h.dot(&self.w2) + &self.b2;
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
        let out_pre = h.dot(&self.w2) + &self.b2;
        let max = out_pre.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = out_pre.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        let out = exp.mapv(|v| v / sum);

        let delta_out = &out - &t;
        let grad_w2 = h
            .insert_axis(Axis(1))
            .dot(&delta_out.clone().insert_axis(Axis(0)));
        let grad_b2 = delta_out.clone();
        let delta_h = delta_out.dot(&self.w2.t()) * h_pre.mapv(|v| 1.0 - v.tanh().powi(2));
        let grad_w1 = x
            .insert_axis(Axis(1))
            .dot(&delta_h.clone().insert_axis(Axis(0)));
        let grad_b1 = delta_h;

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
        npz.finish()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut npz = NpzReader::new(file)?;
        let sample_rate: ndarray::Array1<i64> = npz.by_name("sample_rate.npy")?;
        let bits: ndarray::Array1<i64> = npz.by_name("bits.npy")?;
        Ok(Self {
            w1: npz.by_name("w1.npy")?,
            b1: npz.by_name("b1.npy")?,
            w2: npz.by_name("w2.npy")?,
            b2: npz.by_name("b2.npy")?,
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_roundtrip() {
        let value: i16 = -1234;
        let bits = i16_to_bits(value);
        let result = bits_to_i16(&bits);
        assert_eq!(value, result);
    }
}
