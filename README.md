# StreamZ

StreamZ is a lightweight prototype for handling real time voice data and processing it with a neural network.  The current version focuses on speaker identification rather than audio reconstruction.

## What the Program Achieves

StreamZ demonstrates a very small audio classification pipeline built entirely in
Rust. The application shows how short WAV or MP3 recordings can be used to train a
feed‑forward neural network and how that network can then recognise a speaker in
new audio samples. It packages the recorded training data and the resulting
model weights so that the classifier can be reused in subsequent runs. While the
network architecture is intentionally simple, the project provides a clear
example of end‑to‑end speaker recognition without relying on large external
libraries.

## Project Goal

The aim of this repository is to experiment with real‑time voice streaming and
neural‑network based speaker recognition.  By building a small Rust
application around a simple network, StreamZ demonstrates how microphone input
can be captured, converted into training samples and then classified.  The code
serves as a starting point for more advanced experiments in real‑time audio
analysis or future work on speech reconstruction.

## Features

- `MIMOStream` simulator generating bit vectors.
- `SimpleNeuralNet` with softmax output for multi speaker classification.
- Samples are normalized to floating point values before neural processing.
- Training samples can optionally be saved as `.wav` files for later use.
- Model weights are stored in an `npz` file so the network can be reused between runs.
- Noise gate threshold adjustable in real time using the Up and Down arrow keys.
- MP3 input files are automatically decoded to WAV samples.
- Unknown speakers can be detected using a confidence threshold.
- The output layer is statically sized for up to 153 speakers.

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

The program listens to your microphone and prints the predicted speaker for short
windows of speech.  On first run you will be prompted to record a few example
sentences for each speaker.  These samples are used to train the network and are
written to `.wav` files along with an `model.npz` weight file.  Subsequent runs
load the saved model automatically. Each speaker's training files are stored
inside `model.npz` so the association between filenames and speakers can be
reused later. The model file also stores each speaker's weights and biases
individually using names like `w1`/`b1`, `w2`/`b2` and so on. The network
preallocates space for 153 speakers and keeps track of how many are trained.
Whenever a new speaker is recognised the next slot is initialised and saved
immediately so training across multiple runs preserves every speaker's
parameters.
During streaming you can press the Up and Down arrow keys to raise or lower the
noise gate threshold. The current level is printed each time you adjust it.

Training file paths are listed in `train_files.txt`. Entries may initially
contain only a path. After a model is trained or loaded, running the program
updates this file by appending the detected speaker number to each path.

Starting from v0.2 the training routine no longer requires all audio files to
share the same sample rate. Every file is analysed and trained individually and
the network updates its stored sample rate before processing the next file. This
makes it possible to mix recordings taken at different rates in a single
training session.

## License

This repository is released under the Creative Commons Zero v1.0 Universal license. See [LICENSE](LICENSE) for the full text.
