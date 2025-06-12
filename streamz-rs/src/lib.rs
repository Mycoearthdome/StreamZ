use hound::{WavReader, WavWriter};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::error::Error;
use tokio::time::{sleep, Duration};
use rodio;

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
        Self {
            w1: Array2::zeros((input, hidden)),
            b1: Array1::zeros(hidden),
            w2: Array2::zeros((hidden, output)),
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
        let grad_w2 = h.insert_axis(Axis(1)).dot(&delta_out.clone().insert_axis(Axis(0)));
        let grad_b2 = delta_out.clone();
        let delta_h = delta_out.dot(&self.w2.t()) * h_pre.mapv(|v| 1.0 - v.tanh().powi(2));
        let grad_w1 = x.insert_axis(Axis(1)).dot(&delta_h.clone().insert_axis(Axis(0)));
        let grad_b1 = delta_h;

        self.w2 -= &(grad_w2 * lr);
        self.b2 -= &(grad_b2 * lr);
        self.w1 -= &(grad_w1 * lr);
        self.b1 -= &(grad_b1 * lr);
    }
}

/// Process an input WAV file and write the processed samples to the output path
pub fn process_wav(input: &str, output: &str, net: &SimpleNeuralNet) -> Result<(), Box<dyn Error>> {
    let mut reader = WavReader::open(input)?;
    let spec = reader.spec();
    let mut writer = WavWriter::create(output, spec)?;

    for sample in reader.samples::<i16>() {
        let sample = sample?;
        let bits = i16_to_bits(sample);
        let out_bits = net.forward(&bits);
        let out_sample = bits_to_i16(&out_bits);
        writer.write_sample(out_sample)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Continuously read bits from a `MIMOStream`, train the network and play the output
pub async fn live_stream(stream: &MIMOStream, net: &mut SimpleNeuralNet) -> Result<(), Box<dyn Error>> {
    let (_out_stream, handle) = rodio::OutputStream::try_default()?;
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
