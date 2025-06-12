use hound::{WavReader, WavWriter};
use ndarray::{Array1, Array2};
use std::error::Error;

/// Convert a 16-bit sample to a vector of bits represented as f32 values (0.0 or 1.0)
fn i16_to_bits(val: i16) -> [f32; 16] {
    let mut bits = [0.0f32; 16];
    for i in 0..16 {
        bits[i] = if (val >> i) & 1 == 1 { 1.0 } else { 0.0 };
    }
    bits
}

/// Convert a vector of bits back into a 16-bit sample
fn bits_to_i16(bits: &[f32]) -> i16 {
    let mut value: i16 = 0;
    for i in 0..16 {
        if bits[i] > 0.5 {
            value |= 1 << i;
        }
    }
    value
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
