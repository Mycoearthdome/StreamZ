# StreamZ

StreamZ is a lightweight prototype for handling Multiple Input Multiple Output (MIMO) data streams and feeding them into a neural network in real time. The original Python example simulated 5G bit vectors for a small feed-forward network.

The project now includes a Rust implementation capable of processing voice data from WAV files at the bit-stream level. The Rust neural network reads an input WAV, runs the audio through a simple network and writes a new WAV file that can be sent back to a smartphone.

## Features

- Asynchronous `MIMOStream` class that emulates receiving and sending bit vectors (Python).
- `SimpleNeuralNet` implemented with NumPy for quick experimentation.
- Rust library `streamz-rs` for reading WAV files, passing the audio through a small neural network and writing a new WAV file.
- Example Python script and Rust example showing real-time processing.

## Requirements

- Python 3.8+ with NumPy for the original demo.
- Rust 1.70+ and Cargo for the `streamz-rs` crate.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Build the Rust example with:

```bash
cargo run --example process_file --manifest-path streamz-rs/Cargo.toml
```

## Usage

Run the original Python simulation:

```bash
python streamz/mimo_nn.py
```

Run the Rust example to process a WAV file:

```bash
cargo run --example process_file --manifest-path streamz-rs/Cargo.toml
```

The Python script prints randomly generated bits, while the Rust example reads `input.wav` and saves a processed file named `output.wav`.

## License

This repository is released under the Creative Commons Zero v1.0 Universal license. See [LICENSE](LICENSE) for the full text.
