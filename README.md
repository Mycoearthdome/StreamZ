# StreamZ

StreamZ is a small Rust application that trains and executes a simple neural network to classify short MP3 (or WAV) recordings by speaker.  The project demonstrates how to read raw audio, convert it into FFT feature windows and incrementally learn new speakers from a list of files.

## Features

- Loads MP3 or 16‑bit WAV files and automatically converts them into normalised FFT feature windows.
- Feed‑forward neural network (`SimpleNeuralNet`) with one hidden layer and softmax output.
- Training files and their assigned class numbers are stored in `train_files.txt` so that additional runs continue learning from where you left off.
- Model weights, sample rate and other metadata are saved in `model.npz` for reuse between sessions.
- Unlabelled files are compared against existing speakers using a confidence threshold (default `0.8`); low confidence will create a new speaker entry automatically. The threshold can be overridden with `--threshold <value>` when running the program.
- Works with recordings that use different sample rates and can grow to
  handle any number of speakers over time.
- A helper function `identify_speaker_list` returns all detected speakers in
  a recording based on per-window predictions.

## Installation

1. Install [Rust](https://www.rust-lang.org/tools/install) 1.70 or newer.
2. Clone this repository and build the project:

```bash
cd streamz-rs
cargo build --release
```

This produces the `StreamZ` binary in `target/release/`.

## Usage

Prepare `train_files.txt` with paths to your training clips.  Each line may optionally contain a comma and a numeric speaker index.  For example:

```
audio/alice1.mp3,0
audio/bob1.mp3,1
unlabelled.mp3
```

Run the classifier:

```bash
./target/release/StreamZ
```

During the run every file listed in `train_files.txt` is processed.  If a line lacks a speaker label the program will attempt to match it against previously learned voices and will append the chosen label to the file.  Newly discovered speakers are added to the model automatically.  The updated model and list of files are written back to disk at the end of the run.

## License

This project is released under the Creative Commons Zero v1.0 Universal license.  See [LICENSE](LICENSE) for details.
