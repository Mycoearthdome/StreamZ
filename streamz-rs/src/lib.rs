use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rodio;
use std::error::Error;
use std::sync::{Arc, Mutex};
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::enable_raw_mode;
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

/// Convert a 16-bit sample to a vector of bits represented as f32 values (0.0 or 1.0)
pub fn i16_to_bits(val: i16) -> [f32; 16] {
    let mut bits = [0.0f32; 16];
    for i in 0..16 {
        bits[i] = if (val >> i) & 1 == 1 { 1.0 } else { 0.0 };
    }
    bits
}

/// Convert a vector of bits back into a 16-bit sample
pub fn bits_to_i16(bits: &[f32]) -> i16 {
    let mut value: i16 = 0;
    for i in 0..16 {
        if bits[i] > 0.5 {
            value |= 1 << i;
        }
    }
    value
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

    let stream = input.build_input_stream(
        &config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            buffer_cb.lock().unwrap().extend_from_slice(data);
        },
        err_fn,
        None,
    )?;

    stream.play()?;
    std::thread::sleep(Duration::from_secs(duration_secs));
    drop(stream);
    let samples = buffer.lock().unwrap().clone();
    Ok(samples)
}

/// Pre-train the network with a slice of `i16` samples.
pub fn pretrain_network(net: &mut SimpleNeuralNet, samples: &[i16], epochs: usize, lr: f32) {
    for _ in 0..epochs {
        for &sample in samples {
            let bits = i16_to_bits(sample);
            net.train(&bits, &bits, lr);
        }
    }
}

/// Record audio for a sentence using a simple silence detector.
pub fn record_sentence(prompt: &str) -> Result<Vec<i16>, Box<dyn Error>> {
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    const AMBIENT_SAMPLES: usize = 44100;

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

    let ambient_sum_cb = ambient_sum.clone();
    let ambient_count_cb = ambient_count.clone();
    let has_baseline_cb = has_baseline.clone();
    let started_cb = started.clone();
    let silence_count_cb = silence_count.clone();
    let finished_cb = finished.clone();

    let err_fn = |err| eprintln!("Stream error: {}", err);

    let stream = input.build_input_stream(
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
                if !started_cb.load(Ordering::Relaxed) {
                    if (sample.abs() as f32) > baseline * 1.2 {
                        started_cb.store(true, Ordering::Relaxed);
                        buffer_cb.lock().unwrap().push(sample);
                    }
                    continue;
                }

                buffer_cb.lock().unwrap().push(sample);
                if (sample.abs() as f32) > baseline * 1.2 {
                    silence_count_cb.store(0, Ordering::Relaxed);
                } else {
                    let c = silence_count_cb.fetch_add(1, Ordering::Relaxed) + 1;
                    if c > sample_rate / 4 {
                        finished_cb.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;
    while !finished.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(stream);
    let result = { buffer.lock().unwrap().clone() };
    Ok(result)
}

/// Record and train the network on a list of prompt sentences.
pub fn train_on_sentences(net: &mut SimpleNeuralNet, sentences: &[&str]) -> Result<(), Box<dyn Error>> {
    for &s in sentences {
        let samples = record_sentence(s)?;
        pretrain_network(net, &samples, 1, 0.001);
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

/// Simple feed-forward neural network operating on bit vectors
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

    /// Forward pass on a slice of f32 bits
    pub fn forward(&self, bits: &[f32]) -> Vec<f32> {
        let x = Array1::from_vec(bits.to_vec());
        let h = (x.dot(&self.w1) + &self.b1).mapv(|v| v.tanh());
        let out = h.dot(&self.w2) + &self.b2;
        out.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }).to_vec()
    }

    /// Single-step training using mean squared error and a simple gradient
    pub fn train(&mut self, bits: &[f32], target: &[f32], lr: f32) {
        let x = Array1::from_vec(bits.to_vec());
        let t = Array1::from_vec(target.to_vec());
        let h_pre = x.dot(&self.w1) + &self.b1;
        let h = h_pre.mapv(|v| v.tanh());
        let out_pre = h.dot(&self.w2) + &self.b2;
        let out = out_pre.mapv(|v| v.tanh());

        let error = &out - &t;
        let delta_out = error * out_pre.mapv(|v| 1.0 - v.tanh().powi(2));
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
        let out_bits = net.forward(&bits);
        net.train(&bits, &bits, 0.001);
        let sample = bits_to_i16(&out_bits);
        let buffer = rodio::buffer::SamplesBuffer::new(1, 44100, vec![sample]);
        sink.append(buffer);
    }
}

/// Capture audio from the default microphone, process it with the network and play the result.
/// The stream is passed directly to the neural network without altering pitch or tone.
pub fn live_mic_stream(net: Arc<Mutex<SimpleNeuralNet>>) -> Result<(), Box<dyn Error>> {
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

    let output = host
        .default_output_device()
        .ok_or("No output device available")?;
    let (_out_stream, handle) = rodio::OutputStream::try_from_device(&output)?;
    let sink = rodio::Sink::try_new(&handle)?;

    let sample_rate = SAMPLE_RATE;
    let channels = CHANNELS;

    let err_fn = |err| eprintln!("Stream error: {}", err);

    println!("Recording your voice for 3 seconds...");
    let samples = record_voice_sample(3)?;
    {
        let mut net_lock = net.lock().unwrap();
        pretrain_network(&mut net_lock, &samples, 1, 0.001);
    }
    println!("Initial training complete.");

    const PROMPTS: [&str; 15] = [
        "The quick brown fox jumps over the lazy dog.",
        "She had your dark suit in greasy wash water all year.",
        "We promptly judged antique ivory buckles for the next prize.",
        "A mad boxer shot a quick, gloved jab to the jaw of his dizzy opponent.",
        "Jack quietly moved up front and seized the big ball of wax.",
        "The job requires extra pluck and zeal from every young wage earner.",
        "Just keep examining every low bid quoted for zinc etchings.",
        "Grumpy wizards make toxic brew for the evil queen and jack.",
        "Many voice match tools rely on distinct phonemes and cadence.",
        "Voices differ in pitch, tone, rhythm, and pronunciation details.",
        "Unique vocal prints can identify speakers with surprising accuracy.",
        "Carefully crafted data improves the precision of speaker models.",
        "Every human voice reveals subtle biological and linguistic traits.",
        "He spoke with calm confidence, barely raising his tone.",
        "Different accents often affect vowel shaping and syllable timing.",
    ];

    {
        let mut net_lock = net.lock().unwrap();
        train_on_sentences(&mut net_lock, &PROMPTS)?;
    }

    const AMBIENT_SAMPLES: usize = 44100; // about one second at 44.1kHz
    let noise_mult = Arc::new(Mutex::new(1.2f32));

    // Spawn thread to listen for arrow key input and adjust noise gate
    {
        let noise_mult_ctrl = noise_mult.clone();
        std::thread::spawn(move || {
            enable_raw_mode().ok();
            loop {
                if event::poll(Duration::from_millis(100)).unwrap() {
                    if let Event::Key(k) = event::read().unwrap() {
                        match k.code {
                            KeyCode::Up => {
                                let mut m = noise_mult_ctrl.lock().unwrap();
                                *m += 0.1;
                                println!("Noise gate multiplier: {:.2}", *m);
                            }
                            KeyCode::Down => {
                                let mut m = noise_mult_ctrl.lock().unwrap();
                                *m = (*m - 0.1).max(0.0);
                                println!("Noise gate multiplier: {:.2}", *m);
                            }
                            _ => {}
                        }
                    }
                }
            }
        });
    }

    let stream = {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let net_clone = net.clone();
        let noise_mult_cb = noise_mult.clone();
        let ambient_sum = Arc::new(Mutex::new(0f32));
        let ambient_count = Arc::new(AtomicUsize::new(0));
        let has_baseline = Arc::new(AtomicBool::new(false));
        let ambient_sum_cb = ambient_sum.clone();
        let ambient_count_cb = ambient_count.clone();
        let has_baseline_cb = has_baseline.clone();

        input.build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut output = Vec::with_capacity(data.len());
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
                    let mult = *noise_mult_cb.lock().unwrap();
                    if (sample.abs() as f32) < baseline * mult {
                        continue;
                    }
                    let mut net = net_clone.lock().unwrap();
                    let bits = i16_to_bits(sample);
                    let out_bits = net.forward(&bits);
                    net.train(&bits, &bits, 0.001);
                    let out_sample = bits_to_i16(&out_bits);
                    output.push(out_sample);
                }
                if !output.is_empty() {
                    let buffer = rodio::buffer::SamplesBuffer::new(
                        channels,
                        sample_rate,
                        output.clone(),
                    );
                    sink.append(buffer);
                }
            },
            err_fn,
            None,
        )?
    };

    stream.play()?;
    loop {
        std::thread::sleep(Duration::from_millis(100));
    }
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
