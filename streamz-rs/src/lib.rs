use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rodio;
use std::error::Error;
use std::fs::File;
use std::sync::{Arc, Mutex};
use ndarray_npy::{NpzReader, NpzWriter};
use hound;
use tokio::time::{sleep, Duration};

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 1;

#[allow(dead_code)]
fn select_audio_host() -> cpal::Host {
    #[cfg(target_os = "linux")]
    {
        use cpal::{host_from_id, HostId};
        // Use ALSA directly on Linux and fall back to the default host if
        // it cannot be created.
        host_from_id(HostId::Alsa).unwrap_or_else(|_| cpal::default_host())
    }
    #[cfg(not(target_os = "linux"))]
    {
        cpal::default_host()
    }
}

/// Return the name of the audio backend in use.
/// On Linux this library always uses ALSA.
fn detect_audio_sink() -> &'static str {
    "ALSA"
}

/// Build an input stream and convert incoming samples to i16
fn build_input_stream_i16<F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut data_cb: F,
    mut err_cb: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, Box<dyn Error>>
where
    F: FnMut(&[i16], &cpal::InputCallbackInfo) + Send + 'static,
{
    use cpal::{Sample, SampleFormat};
    let format = device.default_input_config()?.sample_format();
    match format {
        SampleFormat::I16 => Ok(device.build_input_stream(config, data_cb, err_cb, None)?),
        SampleFormat::U16 => Ok(device.build_input_stream(
            config,
            move |data: &[u16], info| {
                let converted: Vec<i16> = data.iter().map(|&s| s.to_sample::<i16>()).collect();
                data_cb(&converted, info);
            },
            err_cb,
            None,
        )?),
        SampleFormat::F32 => Ok(device.build_input_stream(
            config,
            move |data: &[f32], info| {
                let converted: Vec<i16> = data.iter().map(|&s| s.to_sample::<i16>()).collect();
                data_cb(&converted, info);
            },
            err_cb,
            None,
        )?),
        other => Err(format!("Unsupported input sample format: {:?}", other).into()),
    }
}
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

/// Record a short voice sample from the default microphone and return raw `i16` samples.
pub fn record_voice_sample(duration_secs: u64) -> Result<Vec<i16>, Box<dyn Error>> {
    use std::sync::{Arc, Mutex};
    let host = select_audio_host();
    let input = host
        .default_input_device()
        .ok_or("No input device available")?;
    let config = cpal::StreamConfig {
        channels: CHANNELS,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let buffer_cb = buffer.clone();
    let err_fn = |err| eprintln!("Stream error: {}", err);
    let stream = build_input_stream_i16(
        &input,
        &config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            buffer_cb.lock().unwrap().extend_from_slice(data);
        },
        err_fn,
    )?;

    stream.play()?;
    std::thread::sleep(Duration::from_secs(duration_secs));
    drop(stream);
    let samples = buffer.lock().unwrap().clone();
    Ok(samples)
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
    for _ in 0..epochs {
        for &sample in samples {
            let val = i16_to_f32(sample);
            net.train(&[val], &target, lr);
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

/// Train the network using a list of `(path, class)` tuples containing WAV files.
pub fn train_from_files(
    net: &mut SimpleNeuralNet,
    files: &[(&str, usize)],
    num_speakers: usize,
    epochs: usize,
    lr: f32,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..epochs {
        for &(path, class) in files {
            let samples = load_wav_samples(path)?;
            pretrain_network(net, &samples, class, num_speakers, 1, lr);
        }
    }
    Ok(())
}

/// Record audio for a sentence using a simple silence detector.
pub fn record_sentence(prompt: &str, save_path: Option<&str>) -> Result<Vec<i16>, Box<dyn Error>> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    const AMBIENT_SAMPLES: usize = 44100;
    const MAX_RECORD_SECS: usize = 8;

    println!("Please say: \"{}\"", prompt);

    let host = select_audio_host();
    let input = host
        .default_input_device()
        .ok_or("No input device available")?;
    let config = cpal::StreamConfig {
        channels: CHANNELS,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };
    let sample_rate = SAMPLE_RATE as usize;

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let buffer_cb = buffer.clone();

    let ambient_sum = Arc::new(Mutex::new(0f32));
    let ambient_count = Arc::new(AtomicUsize::new(0));
    let has_baseline = Arc::new(AtomicBool::new(false));

    let started = Arc::new(AtomicBool::new(false));
    let silence_count = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicBool::new(false));
    let sample_count = Arc::new(AtomicUsize::new(0));

    let ambient_sum_cb = ambient_sum.clone();
    let ambient_count_cb = ambient_count.clone();
    let has_baseline_cb = has_baseline.clone();
    let started_cb = started.clone();
    let silence_count_cb = silence_count.clone();
    let finished_cb = finished.clone();
    let sample_count_cb = sample_count.clone();

    let filter_prev = Arc::new(Mutex::new(0f32));
    let filter_prev_cb = filter_prev.clone();

    let err_fn = |err| eprintln!("Stream error: {}", err);

    let stream = build_input_stream_i16(
        &input,
        &config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            if !has_baseline_cb.load(Ordering::Relaxed) {
                let mut sum = ambient_sum_cb.lock().unwrap();
                for &sample in data {
                    *sum += sample.abs() as f32;
                    if ambient_count_cb.fetch_add(1, Ordering::Relaxed) >= AMBIENT_SAMPLES {
                        *sum /= AMBIENT_SAMPLES as f32;
                        has_baseline_cb.store(true, Ordering::Relaxed);
                        println!("Ambient noise level: {:.2}", *sum);
                        break;
                    }
                }
                return;
            }

            let baseline = *ambient_sum_cb.lock().unwrap();
            for &sample in data {
                let processed = subtract_baseline(sample, baseline);
                let filtered = {
                    let mut prev = filter_prev_cb.lock().unwrap();
                    low_pass_filter(processed, &mut *prev, 0.05)
                };
                let processed_i16 = filtered as i16;
                if !started_cb.load(Ordering::Relaxed) {
                    if filtered.abs() > baseline * 0.3 {
                        started_cb.store(true, Ordering::Relaxed);
                        sample_count_cb.store(0, Ordering::Relaxed);
                        buffer_cb.lock().unwrap().push(processed_i16);
                    }
                    continue;
                }

                if filtered.abs() < baseline * 0.3 {
                    let c = silence_count_cb.fetch_add(1, Ordering::Relaxed) + 1;
                    if c > sample_rate / 4 {
                        finished_cb.store(true, Ordering::Relaxed);
                        break;
                    }
                    continue;
                }
                silence_count_cb.store(0, Ordering::Relaxed);
                buffer_cb.lock().unwrap().push(processed_i16);
                sample_count_cb.fetch_add(1, Ordering::Relaxed);
                if sample_count_cb.load(Ordering::Relaxed) > sample_rate * MAX_RECORD_SECS {
                    finished_cb.store(true, Ordering::Relaxed);
                    break;
                }
            }
        },
        err_fn,
    )?;

    stream.play()?;
    while !finished.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(stream);
    let result = { buffer.lock().unwrap().clone() };
    if let Some(path) = save_path {
        let spec = hound::WavSpec {
            channels: CHANNELS,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for s in &result {
            writer.write_sample(*s)?;
        }
        writer.finalize()?;
    }
    Ok(result)
}

/// Record and train the network on a list of prompt sentences.
/// Record each prompt sentence and train the network.
/// The full list is cycled `cycles` times before returning.
pub fn train_on_sentences(
    net: &mut SimpleNeuralNet,
    sentences: &[(&str, usize)],
    num_speakers: usize,
    cycles: usize,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..cycles {
        for (idx, &(s, class)) in sentences.iter().enumerate() {
            let filename = format!("speaker{}_{}.wav", class, idx);
            let samples = record_sentence(s, Some(&filename))?;
            pretrain_network(net, &samples, class, num_speakers, 1, 0.001);
        }
    }
    println!("Sentence training complete.");
    Ok(())
}

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
        }
    }

    pub fn output_size(&self) -> usize {
        self.b2.len()
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
        npz.finish()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut npz = NpzReader::new(file)?;
        Ok(Self {
            w1: npz.by_name("w1.npy")?,
            b1: npz.by_name("b1.npy")?,
            w2: npz.by_name("w2.npy")?,
            b2: npz.by_name("b2.npy")?,
        })
    }
}

/// Predict the speaker ID from a sample slice using the network
pub fn identify_speaker(net: &SimpleNeuralNet, sample: &[i16]) -> usize {
    let mut sums = vec![0.0f32; net.output_size()];
    for &s in sample {
        let out = net.forward(&[i16_to_f32(s)]);
        for (i, v) in out.iter().enumerate() {
            sums[i] += *v;
        }
    }
    sums
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Continuously read bits from a `MIMOStream`, train the network and play the output
pub async fn live_stream(
    stream: &MIMOStream,
    net: &mut SimpleNeuralNet,
) -> Result<(), Box<dyn Error>> {
    let output = select_audio_host()
        .default_output_device()
        .ok_or("No output device available")?;
    println!("Detected audio backend: {}", detect_audio_sink());
    let (_out_stream, handle) = rodio::OutputStream::try_from_device(&output)?;
    let sink = rodio::Sink::try_new(&handle)?;
    loop {
        let bits = stream.get_input_bits().await;
        let out = net.forward(&bits);
        net.train(&bits, &bits, 0.001);
        let thresh: Vec<f32> = out
            .iter()
            .map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let sample = bits_to_i16(&thresh);
        let buffer = rodio::buffer::SamplesBuffer::new(1, 44100, vec![sample]);
        sink.append(buffer);
    }
}

/// Capture audio from the default microphone, process it with the network and play the result.
/// The stream is passed directly to the neural network without altering pitch or tone.
pub fn live_mic_stream(net: Arc<Mutex<SimpleNeuralNet>>, num_speakers: usize) -> Result<(), Box<dyn Error>> {
    let host = select_audio_host();
    println!("Detected audio backend: {}", detect_audio_sink());
    let input = host
        .default_input_device()
        .ok_or("No input device available")?;
    let config = cpal::StreamConfig {
        channels: CHANNELS,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let err_fn = |err| eprintln!("Stream error: {}", err);

    println!("Beginning training. Please read each sentence after the prompt.");

    const PROMPTS: [(&str, usize); 4] = [
        ("Speaker A one", 0),
        ("Speaker A two", 0),
        ("Speaker B one", 1),
        ("Speaker B two", 1),
    ];

    const SENTENCE_CYCLES: usize = 1;

    {
        let mut net_lock = net.lock().unwrap();
        train_on_sentences(&mut net_lock, &PROMPTS, num_speakers, SENTENCE_CYCLES)?;
        net_lock.save("model.npz").ok();
    }

    const AMBIENT_SAMPLES: usize = 44100; // about one second at 44.1kHz
    let noise_mult = Arc::new(Mutex::new(1.2f32));
    let filter_prev = Arc::new(Mutex::new(0f32));
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));

    let stream = {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let net_clone = net.clone();
        let noise_mult_cb = noise_mult.clone();
        let filter_prev_cb = filter_prev.clone();
        let ambient_sum = Arc::new(Mutex::new(0f32));
        let ambient_count = Arc::new(AtomicUsize::new(0));
        let has_baseline = Arc::new(AtomicBool::new(false));
        let ambient_sum_cb = ambient_sum.clone();
        let ambient_count_cb = ambient_count.clone();
        let has_baseline_cb = has_baseline.clone();
        let audio_buffer_cb = audio_buffer.clone();

        build_input_stream_i16(
            &input,
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                if !has_baseline_cb.load(Ordering::Relaxed) {
                    let mut sum = ambient_sum_cb.lock().unwrap();
                    for &sample in data {
                        *sum += sample.abs() as f32;
                        if ambient_count_cb.fetch_add(1, Ordering::Relaxed) >= AMBIENT_SAMPLES {
                            *sum /= AMBIENT_SAMPLES as f32;
                            has_baseline_cb.store(true, Ordering::Relaxed);
                            println!("Ambient noise level: {:.2}", *sum);
                            break;
                        }
                    }
                    return;
                }

                let baseline = *ambient_sum_cb.lock().unwrap();
                for &sample in data {
                    let processed = subtract_baseline(sample, baseline);
                    let filtered = {
                        let mut prev = filter_prev_cb.lock().unwrap();
                        low_pass_filter(processed, &mut *prev, 0.05)
                    };
                    let mult = *noise_mult_cb.lock().unwrap();
                    if filtered.abs() < baseline * mult {
                        continue;
                    }
                    let mut buf = audio_buffer_cb.lock().unwrap();
                    buf.push(filtered as i16);
                    if buf.len() >= 2048 {
                        let slice = buf.clone();
                        buf.clear();
                        let speaker = {
                            let net = net_clone.lock().unwrap();
                            identify_speaker(&net, &slice)
                        };
                        println!("Speaker: {}", speaker);
                    }
                }
            },
            err_fn,
        )?
    };

    stream.play()?;
    enable_raw_mode()?;
    loop {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    match k.code {
                        KeyCode::Up => {
                            let mut m = noise_mult.lock().unwrap();
                            *m += 0.1;
                            println!("Noise gate multiplier: {:.2}", *m);
                        }
                        KeyCode::Down => {
                            let mut m = noise_mult.lock().unwrap();
                            *m = (*m - 0.1).max(0.0);
                            println!("Noise gate multiplier: {:.2}", *m);
                        }
                        KeyCode::Esc => {
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    disable_raw_mode()?;
    drop(stream);
    Ok(())
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
