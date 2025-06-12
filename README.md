# StreamZ

StreamZ is a lightweight prototype for handling Multiple Input Multiple Output (MIMO) data streams and processing them with a neural network in real time.

The project focuses on a Rust implementation capable of processing voice data from WAV files at the bit-stream level. The neural network reads an input WAV, runs the audio through a simple network and either writes a new WAV file or streams the result directly to your speakers.

## Features

- `MIMOStream` simulator generating bit vectors.
- `SimpleNeuralNet` for quick experimentation.
- Rust library `streamz-rs` for reading WAV files, passing the audio through a small neural network and writing or streaming the output.
- Example programs demonstrating file processing and live streaming.

## Requirements

- Rust 1.70+ and Cargo.

Build the examples with:

```bash
cargo run --example process_file --manifest-path streamz-rs/Cargo.toml
cargo run --example live_stream --manifest-path streamz-rs/Cargo.toml
```

## Usage

Run the file processing example:

```bash
cargo run --example process_file --manifest-path streamz-rs/Cargo.toml
```

Run the live streaming example:

```bash
cargo run --example live_stream --manifest-path streamz-rs/Cargo.toml
```

The examples read `input.wav` and either save a processed file named `output.wav` or play the output continuously.

## License

This repository is released under the Creative Commons Zero v1.0 Universal license. See [LICENSE](LICENSE) for the full text.
