# StreamZ

StreamZ is a lightweight prototype for handling Multiple Input Multiple Output (MIMO) data streams and processing them with a neural network in real time.

The project now focuses on streaming voice data directly from your microphone. Audio is captured live, passed through a small neural network and streamed back to your speakers.

## Features

- `MIMOStream` simulator generating bit vectors.
- `SimpleNeuralNet` for quick experimentation.
- Rust library `streamz-rs` for live microphone streaming through a small neural network.
- Automatically selects the available PulseAudio or ALSA sink on Linux.
- Example program demonstrating live streaming from the microphone.

## Requirements

- Rust 1.70+ and Cargo.

Build the example with:

```bash
cargo run --example live_stream --manifest-path streamz-rs/Cargo.toml
```

## Usage

Run the live streaming example:

```bash
cargo run --example live_stream --manifest-path streamz-rs/Cargo.toml
```

The example listens to your microphone and plays the processed signal continuously.
On Linux, the library automatically chooses PulseAudio or ALSA depending on what is available.

## License

This repository is released under the Creative Commons Zero v1.0 Universal license. See [LICENSE](LICENSE) for the full text.
