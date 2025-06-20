# StreamZ

StreamZ is a small Rust application that trains and executes a simple neural network to classify short MP3 or WAV recordings by speaker. It demonstrates how to read raw audio, convert it into MFCC + delta feature windows and incrementally learn new speakers from a list of files.

## Features

- Loads MP3 or 16‑bit WAV files and automatically converts them into normalised MFCC feature windows.
- Feed‑forward neural network (`SimpleNeuralNet`) with one hidden layer and softmax output.
- Training files and their assigned class numbers are stored in `train_files.txt` so that additional runs continue learning from where you left off.
- Model weights, sample rate and other metadata are saved in `model.npz` for reuse between sessions.
- Unlabelled files are compared against existing speakers using a confidence threshold (default `0.8`); low confidence will create a new speaker entry automatically. The threshold can be overridden with `--threshold <value>` when running the program.
- Works with recordings that use different sample rates and can grow to
  handle any number of speakers over time.
- Automatically resamples all audio to 44.1kHz for consistent processing.
- Caches MP3 files as WAV before feature extraction when caching is enabled.
- A helper function `identify_speaker_list` returns all detected speakers in
  a recording based on per-window predictions.
- Optionally caches MFCC + delta feature windows for each audio file as `.npy` in
  `feature_cache/` so repeated runs skip expensive extraction.

## Installation

1. Install [Rust](https://www.rust-lang.org/tools/install) 1.70 or newer.
2. Clone this repository and build the project:

```bash
cd streamz-rs
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This produces the `StreamZ` binary in `target/release/`.

## Usage

Prepare `train_files.txt` with paths to your training clips.  Each line may optionally contain a comma and a numeric speaker index.  For example:

```
audio/alice1.mp3,0
audio/bob1.mp3,1
unlabelled.mp3
```

If `target_files.txt` exists it should contain paths and labels for a
separate evaluation set using the same `path,label` format. When running
with `--eval` these files are loaded from the `cache/` directory if a
matching WAV is found and used to compute accuracy metrics. MP3 files in
either list are automatically converted to `cache/<name>.wav` before
processing. If a neighbouring `.wav` file already exists it is used
instead of creating a cached copy.

After training completes, `target_files.txt` is automatically refreshed with
the cached WAV paths and labels from `train_files.txt`. This lets running the
program with `--eval` use the prepared evaluation list directly.

Run the classifier:

```bash
./target/release/StreamZ [--burn-in-limit <n>] [--max-speakers <n>] [--no-cache-wav] \
                       [--threshold <value>] [--eval] [--eval-split <fraction>] \
                       [--force] [--retrain] [--cluster-embeddings <k>]
```

During the run every file listed in `train_files.txt` is processed.  If a line lacks a speaker label the program will attempt to match it against previously learned voices and will append the chosen label to the file.  Newly discovered speakers are added to the model automatically.  The updated model and list of files are written back to disk at the end of the run.

## Command Line Options

- `--burn-in-limit <n>` sets how many files are treated as the initial burn‑in phase.
- `--max-speakers <n>` caps the number of speaker classes the model will create.
- `--no-cache-wav` prevents writing new cached WAV files but existing ones are reused.
- `--threshold <value>` adjusts the confidence threshold for reusing a known label.
- `--eval` runs the program in evaluation mode instead of updating the model.
- `--eval-split <fraction>` controls what fraction of labelled data is reserved for evaluation when `target_files.txt` is not present (default `0.2`).
- `--force` retrains the model even when a saved model exists.
- `--retrain` behaves like `--force` and can be used without `--eval`.
- `--check-embeddings` loads `model.npz` and prints quality metrics for any embeddings stored in the file. If none are saved it recomputes them from the training data.
- `--cluster-embeddings <k>` groups saved speaker embeddings into `k` clusters and prints the assignments.

## Feature Caching

When an audio file is first processed its MFCC + delta feature windows can be saved
to `feature_cache/<name>.npy`. Subsequent runs load these cached arrays using the
`load_cached_features` helper which avoids recomputing features every time. This
greatly speeds up repeated training or evaluation on the same dataset.

## Training Tips

 - The default training loop now runs for 60 epochs per file with a small
  dropout rate of 20% to reduce overfitting.
- You can adjust the confidence threshold for matching existing speakers using
  `--threshold <value>`. Lower values (e.g. `0.5`) make the program more willing
  to reuse a known speaker label.
- Use `--burn-in-limit <n>` to specify how many files are treated as an
  early burn-in phase with a higher matching threshold. By default around 20%
  of the dataset (between 10 and 50 files) is used.
- `--max-speakers <n>` prevents unbounded growth of speaker classes. The
  default limit is the current number of speakers plus ten.
- `--no-cache-wav` prevents writing new cached WAV files but existing ones are
  reused for faster startup.
- Use `--eval` to measure model accuracy without updating weights.
- `--eval-split <fraction>` sets the portion of labelled data reserved for evaluation when `target_files.txt` is absent.
- `--check-embeddings` loads `model.npz` and prints quality metrics for any embeddings stored in the file. If none are saved it recomputes them from the training data.
- `--cluster-embeddings <k>` groups saved speaker embeddings into `k` clusters and prints the assignments.

## Threaded Components

Several stages now run in parallel using the [`rayon`](https://crates.io/crates/rayon) thread pool.

| Task | Threaded? |
| --- | --- |
| Audio loading | ✅ |
| Pretraining | ✅ |
| Evaluation | ✅ |
| Cosine matching | ✅ |
| Embedding computation | ✅ |

## License

This project is released under the Creative Commons Zero v1.0 Universal license.  See [LICENSE](LICENSE) for details.
