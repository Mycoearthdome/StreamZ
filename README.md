# StreamZ

StreamZ is a lightweight prototype for handling Multiple Input Multiple Output (MIMO) data streams and processing them with a neural network in real time.

The project now focuses on streaming voice data directly from your microphone. Audio is captured live, passed through a small neural network and streamed back to your speakers.

## Features

- `MIMOStream` simulator generating bit vectors.
- `SimpleNeuralNet` for quick experimentation.
- Rust library `streamz-rs` for live microphone streaming through a small neural network.
- Uses ALSA for audio output on Linux.
- Command line program demonstrating live streaming from the microphone.

## Requirements

- Rust 1.70+ and Cargo.
- On Linux, the `libasound2-dev` package is required to build the ALSA backend
  used by the `rodio` and `cpal` crates. Install it via `apt`:

```bash
sudo apt-get install libasound2-dev
```

Build the program with:

```bash
cd streamz-rs
cargo build --release
```

## Usage

Run the live streaming program:

```bash
./target/release/StreamZ
```

The program listens to your microphone and plays the processed signal continuously.
Audio output uses ALSA on Linux.

## License

This repository is released under the Creative Commons Zero v1.0 Universal license. See [LICENSE](LICENSE) for the full text.
