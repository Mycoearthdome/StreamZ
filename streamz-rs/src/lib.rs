// use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
// use crossterm::event::{self, Event, KeyCode, KeyEventKind};
// use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use hound;
use mel_filter::{mel, NormalizationFactor};
use minimp3::{Decoder, Error as Mp3Error, Frame};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_npy::{read_npy, write_npy, NpzReader, NpzWriter};
use rand::seq::SliceRandom;
use rand::Rng;
use rustdct::{DctPlanner, TransformType2And3};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::io::BufReader;
// use rodio;
use once_cell::sync::Lazy;
use rubato::{FftFixedInOut, Resampler};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub const DEFAULT_SAMPLE_RATE: u32 = 44100;
pub const WINDOW_SIZE: usize = 800;
const N_MELS: usize = 26;
pub const MFCC_SIZE: usize = 20;
pub const WITH_DELTAS: bool = true;
pub const FEATURE_SIZE: usize = if WITH_DELTAS {
    MFCC_SIZE * 3
} else {
    MFCC_SIZE
};
/// Default dropout probability applied during training.
pub const DEFAULT_DROPOUT: f32 = 0.2;

/// Whether WAV caching is enabled. This can be toggled at runtime via
/// [`set_wav_cache_enabled`]. Caching is enabled by default so that the
/// same files are not repeatedly converted.
static WAV_CACHE_ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable writing WAV cache files.
pub fn set_wav_cache_enabled(enabled: bool) {
    WAV_CACHE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns `true` if WAV caching is currently enabled.
pub fn wav_cache_enabled() -> bool {
    WAV_CACHE_ENABLED.load(Ordering::Relaxed)
}

/// Shared pool of resamplers keyed by input sample rate.
static RESAMPLERS: Lazy<Mutex<HashMap<u32, Arc<Mutex<FftFixedInOut<f32>>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn get_resampler(from_rate: u32) -> Result<Arc<Mutex<FftFixedInOut<f32>>>, Box<dyn Error>> {
    let mut map = RESAMPLERS.lock().unwrap();
    if let Some(r) = map.get(&from_rate) {
        return Ok(r.clone());
    }
    let resampler =
        FftFixedInOut::<f32>::new(from_rate as usize, DEFAULT_SAMPLE_RATE as usize, 1024, 1)?;
    let arc = Arc::new(Mutex::new(resampler));
    map.insert(from_rate, arc.clone());
    Ok(arc)
}

/// Apply data augmentation to raw i16 samples.
///
/// In addition to random gain and noise, a small random time shift is
/// applied to each recording. This helps the network become more
/// robust to differences in leading/trailing silence.
pub fn augment(samples: &[i16]) -> Vec<i16> {
    let mut rng = rand::thread_rng();
    let noise_level: f32 = rng.gen_range(0.0..0.005);
    let gain: f32 = rng.gen_range(0.95..1.05);
    let shift = rng.gen_range(0..samples.len().min(WINDOW_SIZE));
    let mut out = Vec::with_capacity(samples.len());
    for i in 0..samples.len() {
        let idx = (i + shift) % samples.len();
        let noise: f32 = rng.gen_range(-noise_level..noise_level);
        let val = samples[idx] as f32 * gain + noise * i16::MAX as f32;
        out.push(val.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
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

/// Normalize a vector to unit length using the L2 norm
fn normalize(v: &mut [f32]) {
    let view = ndarray::aview1(v);
    let norm = view.into_par_iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let mut view_mut = ndarray::aview_mut1(v);
        view_mut.par_mapv_inplace(|x| x / norm);
    }
}

/// Convert a raw i16 audio sample to a normalized f32 value in [-1.0, 1.0]
pub fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

/// Downmix multi-channel samples to mono by averaging channels.
pub fn downmix_to_mono(samples: &[i16], channels: usize) -> Vec<i16> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|c| {
            let sum: i32 = c.iter().map(|&s| s as i32).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

/// Resample i16 samples to 44.1kHz using rubato
pub fn resample_to_44100(samples: &[i16], from_rate: u32) -> Result<Vec<i16>, Box<dyn Error>> {
    if from_rate == DEFAULT_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }

    let input: Vec<Vec<f32>> = vec![samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect()];
    let frames_in = input[0].len();
    let frames_out = (frames_in * DEFAULT_SAMPLE_RATE as usize) / from_rate as usize;
    let mut output = vec![vec![0.0f32; frames_out]];

    let resampler_arc = get_resampler(from_rate)?;
    {
        let mut resampler = resampler_arc.lock().unwrap();
        resampler.process_into_buffer(&input, &mut output, None)?;
    }

    Ok(output[0]
        .iter()
        .map(|&s| (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect())
}

/// Compute delta features for a sequence of frames
fn add_deltas(mfcc: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(mfcc.len());
    for i in 0..mfcc.len() {
        let prev = if i > 0 { &mfcc[i - 1] } else { &mfcc[i] };
        let next = if i + 1 < mfcc.len() {
            &mfcc[i + 1]
        } else {
            &mfcc[i]
        };
        let mut delta = Vec::with_capacity(MFCC_SIZE);
        for j in 0..MFCC_SIZE {
            delta.push((next[j] - prev[j]) / 2.0);
        }
        out.push(delta);
    }
    out
}

/// Precomputed FFT, DCT and Mel filters for efficient feature extraction.
pub struct FeatureExtractor {
    fft: Arc<dyn Fft<f32>>,
    dct: Arc<dyn TransformType2And3<f32>>,
    mel_filters: Vec<Vec<f32>>,
}

impl FeatureExtractor {
    /// Create a new extractor with cached transform plans.
    pub fn new() -> Self {
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
        Self {
            fft,
            dct,
            mel_filters,
        }
    }

    /// Extract MFCC features from a slice of samples.
    pub fn extract(&self, samples: &[i16]) -> Vec<Vec<f32>> {
        window_samples_with_plan(samples, &self.mel_filters, &self.fft, &self.dct)
    }
}

thread_local! {
    static EXTRACTOR_TLS: FeatureExtractor = FeatureExtractor::new();
}

/// Run a closure with a thread-local `FeatureExtractor` instance.
pub fn with_thread_extractor<F, R>(f: F) -> R
where
    F: FnOnce(&FeatureExtractor) -> R,
{
    EXTRACTOR_TLS.with(|ex| f(ex))
}

/// Split samples into windows using precomputed transform plans.
fn window_samples_with_plan(
    samples: &[i16],
    mel_filters: &[Vec<f32>],
    fft: &Arc<dyn Fft<f32>>,
    dct: &Arc<dyn TransformType2And3<f32>>,
) -> Vec<Vec<f32>> {
    let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); WINDOW_SIZE];
    let mut base = Vec::new();

    let step = WINDOW_SIZE / 2;
    if samples.len() >= WINDOW_SIZE {
        let mut start = 0;
        while start + WINDOW_SIZE <= samples.len() {
            let chunk = &samples[start..start + WINDOW_SIZE];
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
            coeffs.truncate(MFCC_SIZE);
            base.push(coeffs);

            start += step;
        }
    }

    let deltas = add_deltas(&base);
    let delta2 = add_deltas(&deltas);
    let mut features = Vec::with_capacity(base.len());
    for i in 0..base.len() {
        let mut frame = base[i].clone();
        frame.extend(&deltas[i]);
        frame.extend(&delta2[i]);
        features.push(frame);
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
    batch_size: usize,
    extractor: &FeatureExtractor,
) -> f32 {
    let mut target = vec![0.0f32; num_classes];
    if target_class < num_classes {
        target[target_class] = 1.0;
    }
    let mut rng = rand::thread_rng();
    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for _ in 0..epochs {
        let aug_samples = augment(samples);
        let mut windows = extractor.extract(&aug_samples);
        windows.shuffle(&mut rng);
        for chunk in windows.chunks(batch_size.max(1)) {
            let mut batch = Vec::with_capacity(chunk.len());
            for win in chunk {
                let mut feats = win.clone();
                apply_dropout(&mut feats, dropout);
                let out = net.forward(&feats);
                let loss: f32 = -target
                    .iter()
                    .zip(out.iter())
                    .map(|(t, o)| t * o.max(1e-12).ln())
                    .sum::<f32>();
                total_loss += loss;
                count += 1;
                batch.push(feats);
            }
            net.train_batch(&batch, &target, lr);
        }
    }
    if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    }
}

/// Load all samples from a WAV file.
/// Returns raw interleaved samples, the sample rate and channel count.
pub fn load_wav_samples(path: &str) -> Result<(Vec<i16>, u32, usize), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = hound::WavReader::new(BufReader::new(file))?;
    if reader.spec().bits_per_sample != 16 {
        return Err("Only 16-bit audio supported".into());
    }
    let sample_rate = reader.spec().sample_rate;
    let channels = reader.spec().channels as usize;
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    let samples = samples?;
    Ok((samples, sample_rate, channels))
}

/// Load samples from an MP3 file using the `minimp3` decoder.
/// Returns raw interleaved samples, the sample rate and channel count.
pub fn load_mp3_samples(path: &str) -> Result<(Vec<i16>, u32, usize), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file));
    let mut samples = Vec::new();
    let mut sample_rate = 0u32;
    let mut channels = 1usize;
    loop {
        match decoder.next_frame() {
            Ok(Frame {
                data,
                sample_rate: sr,
                channels: ch,
                ..
            }) => {
                if sample_rate == 0 {
                    sample_rate = sr as u32;
                    channels = ch as usize;
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
    Ok((samples, sample_rate, channels))
}

/// Load audio samples from either a WAV or MP3 file depending on the file
/// extension.
pub fn load_audio_samples(path: &str) -> Result<Vec<i16>, Box<dyn Error>> {
    if path.to_ascii_lowercase().ends_with(".mp3") {
        let cache_dir = Path::new("cache");
        let file_stem = Path::new(path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let cached_path = cache_dir.join(format!("{file_stem}.wav"));

        if cached_path.exists() {
            return load_and_resample_file(cached_path.to_str().unwrap())
                .map(|(_, s)| s)
                .map_err(|e| Box::<dyn Error>::from(e));
        }

        let (_, resampled) = load_and_resample_file(path).map_err(|e| Box::<dyn Error>::from(e))?;

        if wav_cache_enabled() {
            let _ = std::fs::create_dir_all(cache_dir);
            let mut writer = hound::WavWriter::create(
                &cached_path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: DEFAULT_SAMPLE_RATE,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?;
            for sample in &resampled {
                writer.write_sample(*sample)?;
            }
            writer.finalize()?;
        }

        Ok(resampled)
    } else {
        load_and_resample_file(path)
            .map(|(_, s)| s)
            .map_err(|e| Box::<dyn Error>::from(e))
    }
}

/// Read the sample rate and bit depth from an audio file.
/// Supports WAV and MP3 formats.
pub fn audio_metadata(path: &str) -> Result<(u32, u16), Box<dyn Error>> {
    if path.to_ascii_lowercase().ends_with(".mp3") {
        let file = File::open(path)?;
        let mut decoder = Decoder::new(BufReader::new(file));
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

/// Load a file and resample its samples to 44.1kHz.
/// Returns the filename along with resampled samples.
pub fn load_and_resample_file(path: &str) -> Result<(String, Vec<i16>), String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "wav" => {
            let reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
            let spec = reader.spec();
            let channels = spec.channels as usize;
            let samples: Vec<i16> = reader
                .into_samples::<i16>()
                .filter_map(Result::ok)
                .collect();
            let mono = downmix_to_mono(&samples, channels);
            let resampled =
                resample_to_44100(&mono, spec.sample_rate).map_err(|e| e.to_string())?;
            Ok((path.to_string(), resampled))
        }
        "mp3" => {
            let (samples, rate, ch) = load_mp3_samples(path).map_err(|e| e.to_string())?;
            let mono = downmix_to_mono(&samples, ch);
            let resampled = resample_to_44100(&mono, rate).map_err(|e| e.to_string())?;
            Ok((path.to_string(), resampled))
        }
        _ => Err(format!("Unsupported format: {}", path)),
    }
}

/// Load and resample multiple files in parallel.
pub fn batch_resample(paths: &[String]) -> Vec<(String, Vec<i16>)> {
    use rayon::prelude::*;
    paths
        .par_iter()
        .filter_map(|p| match load_and_resample_file(p) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Error in {}: {}", p, e);
                None
            }
        })
        .collect()
}

/// Path for cached feature files corresponding to an audio path.
fn feature_cache_path(path: &str) -> std::path::PathBuf {
    let cache_dir = std::path::Path::new("feature_cache");
    let _ = std::fs::create_dir_all(cache_dir);
    let sanitized = path.replace('/', "_").replace('\\', "_");
    cache_dir.join(format!("{}.npy", sanitized))
}

/// Load cached MFCC features for a file or compute and store them if missing.
pub fn load_cached_features(
    path: &str,
    extractor: &FeatureExtractor,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let cache = feature_cache_path(path);
    if cache.exists() {
        let arr: Array2<f32> = read_npy(&cache)?;
        let mut feats = Vec::with_capacity(arr.nrows());
        for row in arr.outer_iter() {
            feats.push(row.to_vec());
        }
        return Ok(feats);
    }
    let samples = load_audio_samples(path)?;
    let feats = extractor.extract(&samples);
    if !feats.is_empty() {
        let flat: Vec<f32> = feats.iter().flat_map(|v| v.clone()).collect();
        let arr = Array2::from_shape_vec((feats.len(), feats[0].len()), flat)?;
        let _ = write_npy(&cache, &arr);
    }
    Ok(feats)
}

/// Train the network using feature windows instead of raw audio.
pub fn pretrain_from_features(
    net: &mut SimpleNeuralNet,
    windows: &[Vec<f32>],
    target_class: usize,
    num_classes: usize,
    epochs: usize,
    lr: f32,
    dropout: f32,
    batch_size: usize,
) -> f32 {
    let mut target = vec![0.0f32; num_classes];
    if target_class < num_classes {
        target[target_class] = 1.0;
    }
    let mut rng = rand::thread_rng();
    let mut total_loss = 0.0f32;
    let mut count = 0usize;
    for _ in 0..epochs {
        let mut wins: Vec<Vec<f32>> = windows.to_vec();
        wins.shuffle(&mut rng);
        for chunk in wins.chunks(batch_size.max(1)) {
            let mut batch = Vec::with_capacity(chunk.len());
            for win in chunk {
                let mut feats = win.clone();
                apply_dropout(&mut feats, dropout);
                let out = net.forward(&feats);
                let loss: f32 = -target
                    .iter()
                    .zip(out.iter())
                    .map(|(t, o)| t * o.max(1e-12).ln())
                    .sum::<f32>();
                total_loss += loss;
                count += 1;
                batch.push(feats);
            }
            net.train_batch(&batch, &target, lr);
        }
    }
    if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    }
}

/// Train the network using a list of `(path, class)` tuples containing WAV files.
pub fn train_from_files(
    net: Arc<RwLock<SimpleNeuralNet>>,
    files: &[(&str, usize)],
    total_files: usize,
    num_speakers: usize,
    epochs: usize,
    lr: f32,
    dropout: f32,
    batch_size: usize,
    _extractor: &FeatureExtractor,
) -> Result<(), Box<dyn Error>> {
    use indicatif::{ProgressBar, ProgressStyle};
    use rayon::prelude::*;
    println!("Training on {} files individually", total_files);
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}/{len} ETA {eta}")
            .unwrap(),
    );

    let step = AtomicI32::new(0);

    files.par_iter().for_each(|&(path, class)| {
        pb.println(path);

        let samples = match load_and_resample_file(path) {
            Ok((_, s)) => s,
            Err(e) => {
                eprintln!("Skipping {}: {}", path, e);
                pb.inc(1);
                return;
            }
        };

        {
            let mut guard = net.write().unwrap();
            // All audio is resampled to DEFAULT_SAMPLE_RATE and 16 bits
            guard.set_dataset_specs(DEFAULT_SAMPLE_RATE, 16);
        }

        for _ in 0..epochs {
            let lr_scaled = lr * (0.99f32).powi(step.fetch_add(1, Ordering::SeqCst));
            let mut guard = net.write().unwrap();
            with_thread_extractor(|ext| {
                let _ = pretrain_network(
                    &mut *guard,
                    &samples,
                    class,
                    num_speakers,
                    1,
                    lr_scaled,
                    dropout,
                    batch_size,
                    ext,
                );
            });
            guard.record_training_file(class, path);
        }

        pb.inc(1);
    });

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

    /// Access the list of training files for each speaker.
    pub fn file_lists(&self) -> &[Vec<String>] {
        &self.file_lists
    }

    /// Forward pass on a slice of f32 values
    pub fn forward(&self, bits: &[f32]) -> Vec<f32> {
        let x = Array1::from_vec(bits.to_vec());
        let h1 = (x.dot(&self.w1) + &self.b1).mapv(|v| if v > 0.0 { v } else { 0.0 });
        let h2 = (h1.dot(&self.w2) + &self.b2).mapv(|v| v.tanh());
        let w3 = self.w3.slice(s![.., ..self.num_speakers]);
        let b3 = self.b3.slice(s![..self.num_speakers]);
        let out = h2.dot(&w3) + &b3;
        let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = out.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        (exp / sum).to_vec()
    }

    /// Compute an embedding vector from the second hidden layer without the
    /// final softmax classification step.
    pub fn embed(&self, bits: &[f32]) -> Vec<f32> {
        let x = Array1::from_vec(bits.to_vec());
        let h1 = (x.dot(&self.w1) + &self.b1).mapv(|v| if v > 0.0 { v } else { 0.0 });
        let h2 = (h1.dot(&self.w2) + &self.b2).mapv(|v| v.tanh());
        h2.to_vec()
    }

    /// Dimension of the embedding vector returned by [`embed`].
    pub fn embedding_size(&self) -> usize {
        self.w2.ncols()
    }

    /// Single-step training using cross entropy loss
    pub fn train(&mut self, bits: &[f32], target: &[f32], lr: f32) {
        let x = Array1::from_vec(bits.to_vec());
        let t = Array1::from_vec(target.to_vec());
        let h1_pre = x.dot(&self.w1) + &self.b1;
        let h1 = h1_pre.mapv(|v| if v > 0.0 { v } else { 0.0 });
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
            .view()
            .insert_axis(Axis(1))
            .dot(&delta_out.clone().insert_axis(Axis(0)));
        let grad_b3 = delta_out.clone();
        let delta_h2 = delta_out.dot(&w3.t()) * h2.mapv(|v| 1.0 - v * v);
        let grad_w2 = h1
            .insert_axis(Axis(1))
            .dot(&delta_h2.clone().insert_axis(Axis(0)));
        let grad_b2 = delta_h2.clone();
        let delta_h1 =
            delta_h2.dot(&self.w2.t()) * h1_pre.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
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

    /// Train on a batch of feature vectors using the average gradient
    pub fn train_batch(&mut self, batch: &[Vec<f32>], target: &[f32], lr: f32) {
        if batch.is_empty() {
            return;
        }
        let mut grad_w1 = Array2::<f32>::zeros(self.w1.raw_dim());
        let mut grad_b1 = Array1::<f32>::zeros(self.b1.raw_dim());
        let mut grad_w2 = Array2::<f32>::zeros(self.w2.raw_dim());
        let mut grad_b2 = Array1::<f32>::zeros(self.b2.raw_dim());
        let mut grad_w3 = Array2::<f32>::zeros((self.w3.nrows(), self.num_speakers));
        let mut grad_b3 = Array1::<f32>::zeros(self.num_speakers);

        for bits in batch {
            let x = Array1::from_vec(bits.clone());
            let t = Array1::from_vec(target.to_vec());
            let h1_pre = x.dot(&self.w1) + &self.b1;
            let h1 = h1_pre.mapv(|v| if v > 0.0 { v } else { 0.0 });
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
            grad_w3 += &h2
                .view()
                .insert_axis(Axis(1))
                .dot(&delta_out.clone().insert_axis(Axis(0)));
            grad_b3 += &delta_out;
            let delta_h2 = delta_out.dot(&w3.t()) * h2.mapv(|v| 1.0 - v * v);
            grad_w2 += &h1
                .insert_axis(Axis(1))
                .dot(&delta_h2.clone().insert_axis(Axis(0)));
            grad_b2 += &delta_h2;
            let delta_h1 =
                delta_h2.dot(&self.w2.t()) * h1_pre.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            grad_w1 += &x
                .insert_axis(Axis(1))
                .dot(&delta_h1.clone().insert_axis(Axis(0)));
            grad_b1 += &delta_h1;
        }

        let scale = lr / batch.len() as f32;
        {
            let mut w3_mut = self.w3.slice_mut(s![.., ..self.num_speakers]);
            w3_mut -= &(grad_w3 * scale);
        }
        {
            let mut b3_mut = self.b3.slice_mut(s![..self.num_speakers]);
            b3_mut -= &(grad_b3 * scale);
        }
        self.w2 -= &(grad_w2 * scale);
        self.b2 -= &(grad_b2 * scale);
        self.w1 -= &(grad_w1 * scale);
        self.b1 -= &(grad_b1 * scale);
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
pub fn identify_speaker(
    net: &SimpleNeuralNet,
    sample: &[i16],
    extractor: &FeatureExtractor,
) -> usize {
    let mut sums = vec![0.0f32; net.output_size()];
    let windows = extractor.extract(sample);
    for win in windows {
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
    extractor: &FeatureExtractor,
) -> Option<usize> {
    // If there's only a single speaker known, every prediction will trivially
    // have a confidence of 1.0. In that case we should treat the result as
    // "unknown" so new speakers can be added.
    if net.output_size() <= 1 {
        return None;
    }
    let mut sums = vec![0.0f32; net.output_size()];
    let mut count = 0f32;
    let windows = extractor.extract(sample);
    for win in windows {
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
pub fn identify_speaker_list(
    net: &SimpleNeuralNet,
    sample: &[i16],
    threshold: f32,
    extractor: &FeatureExtractor,
) -> Vec<usize> {
    let mut counts = vec![0usize; net.output_size()];
    let windows = extractor.extract(sample);
    for win in windows {
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

/// Compute a robust embedding vector for an audio sample.
///
/// Averaging all window embeddings can flatten subtle vocal cues. This
/// implementation computes the median value per dimension across all
/// windows instead.
pub fn extract_embedding(
    net: &SimpleNeuralNet,
    sample: &[i16],
    extractor: &FeatureExtractor,
) -> Vec<f32> {
    let mut wins = Vec::new();
    let windows = extractor.extract(sample);
    for win in windows {
        wins.push(net.embed(&win));
    }

    if wins.is_empty() {
        return vec![0.0; net.embedding_size()];
    }

    let mut emb = vec![0.0f32; net.embedding_size()];
    for i in 0..net.embedding_size() {
        let mut vals: Vec<f32> = wins.iter().map(|v| v[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = vals.len() / 2;
        emb[i] = if vals.len() % 2 == 0 {
            (vals[mid - 1] + vals[mid]) / 2.0
        } else {
            vals[mid]
        };
    }

    normalize(&mut emb);
    emb
}

/// Compute an embedding from precomputed feature windows.
pub fn extract_embedding_from_features(net: &SimpleNeuralNet, windows: &[Vec<f32>]) -> Vec<f32> {
    let mut wins = Vec::new();
    for win in windows {
        wins.push(net.embed(win));
    }
    if wins.is_empty() {
        return vec![0.0; net.embedding_size()];
    }
    let mut emb = vec![0.0f32; net.embedding_size()];
    for i in 0..net.embedding_size() {
        let mut vals: Vec<f32> = wins.iter().map(|v| v[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = vals.len() / 2;
        emb[i] = if vals.len() % 2 == 0 {
            (vals[mid - 1] + vals[mid]) / 2.0
        } else {
            vals[mid]
        };
    }
    normalize(&mut emb);
    emb
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
///
/// The returned vectors are normalised to unit length so they can be
/// compared using cosine similarity. If any speaker does not yet have
/// recorded samples, a zero vector of the appropriate size will be
/// returned for that entry.
/// Compute mean and standard deviation embeddings for each speaker.
///
/// The returned tuple for each speaker contains `(mean, mean_sim, std_sim)`
/// where `mean` is the average embedding, `mean_sim` is the mean cosine
/// similarity of each training embedding to the mean, and `std_sim` is the
/// standard deviation of those similarities.
pub fn compute_speaker_embeddings(
    net: &SimpleNeuralNet,
    extractor: &FeatureExtractor,
) -> Option<Vec<(Vec<f32>, f32, f32)>> {
    let mut speaker_embeds = Vec::with_capacity(net.output_size());
    for files in net.file_lists.iter().take(net.output_size()) {
        let mut embeds = Vec::new();
        for path in files {
            if let Ok(wins) = load_cached_features(path, extractor) {
                let emb = extract_embedding_from_features(net, &wins);
                embeds.push(emb);
            }
        }
        if embeds.is_empty() {
            speaker_embeds.push((vec![0.0; net.embedding_size()], 0.0, 0.0));
        } else {
            let mut mean = vec![0.0f32; net.embedding_size()];
            for emb in &embeds {
                for (i, v) in emb.iter().enumerate() {
                    mean[i] += *v;
                }
            }
            for v in &mut mean {
                *v /= embeds.len() as f32;
            }
            normalize(&mut mean);

            let mut sims = Vec::new();
            for emb in &embeds {
                sims.push(cosine_similarity(emb, &mean));
            }
            let mean_sim = sims.iter().copied().sum::<f32>() / sims.len() as f32;
            let var = sims
                .iter()
                .map(|s| (*s - mean_sim) * (*s - mean_sim))
                .sum::<f32>()
                / sims.len() as f32;
            let std_sim = var.sqrt();

            speaker_embeds.push((mean, mean_sim, std_sim));
        }
    }
    Some(speaker_embeds)
}

/// Identify a speaker by comparing an embedding against known speaker
/// embeddings using cosine similarity. Returns `Some(index)` if the best
/// similarity exceeds `threshold`.
pub fn identify_speaker_cosine(
    net: &SimpleNeuralNet,
    speaker_embeds: &[(Vec<f32>, f32, f32)],
    sample: &[i16],
    threshold: f32,
    extractor: &FeatureExtractor,
) -> Option<usize> {
    if speaker_embeds.is_empty() {
        return None;
    }
    let emb = extract_embedding(net, sample, extractor);
    let mut best_idx = None;
    let mut best_val = threshold;
    for (i, (mean, mean_sim, std_sim)) in speaker_embeds.iter().enumerate() {
        let sim = cosine_similarity(&emb, mean);
        if sim < (mean_sim - 2.0 * *std_sim) {
            continue;
        }
        let adaptive_factor = if speaker_embeds.len() < 200 { 0.3 } else { 1.0 };
        let dynamic_threshold = mean_sim + *std_sim * adaptive_factor;
        let accepted = sim > 0.35 && (sim > dynamic_threshold || sim > 0.5);
        eprintln!(
            "Sim to speaker {}: {:.4}, dyn_thres: {:.4}, accepted: {}",
            i, sim, dynamic_threshold, accepted
        );
        if accepted && sim > best_val {
            best_val = sim;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Identify a speaker using precomputed feature windows.
pub fn identify_speaker_cosine_feats(
    net: &SimpleNeuralNet,
    speaker_embeds: &[(Vec<f32>, f32, f32)],
    windows: &[Vec<f32>],
    threshold: f32,
) -> Option<usize> {
    if speaker_embeds.is_empty() {
        return None;
    }
    let emb = extract_embedding_from_features(net, windows);
    let mut best_idx = None;
    let mut best_val = threshold;
    for (i, (mean, mean_sim, std_sim)) in speaker_embeds.iter().enumerate() {
        let sim = cosine_similarity(&emb, mean);
        if sim < (mean_sim - 2.0 * *std_sim) {
            continue;
        }
        let adaptive_factor = if speaker_embeds.len() < 200 { 0.3 } else { 1.0 };
        let dynamic_threshold = mean_sim + *std_sim * adaptive_factor;
        let accepted = sim > 0.35 && (sim > dynamic_threshold || sim > 0.5);
        eprintln!(
            "Sim to speaker {}: {:.4}, dyn_thres: {:.4}, accepted: {}",
            i, sim, dynamic_threshold, accepted
        );
        if accepted && sim > best_val {
            best_val = sim;
            best_idx = Some(i);
        }
    }
    best_idx
}
