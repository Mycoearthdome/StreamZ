# StreamZ

StreamZ is a lightweight prototype for handling Multiple Input Multiple Output (MIMO) data streams and feeding them into a neural network in real time. The example code simulates 5G streams that provide bit vectors to a small feed-forward network. The network processes the bits and returns output bits simultaneously.

## Features

- Asynchronous `MIMOStream` class that emulates receiving and sending bit vectors.
- `SimpleNeuralNet` implemented with NumPy for demonstration purposes.
- Example script running a real-time loop where input bits are processed as soon as they arrive.

## Requirements

- Python 3.8+
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulated pipeline:

```bash
python streamz/mimo_nn.py
```

You should see lines printed showing the randomly generated input bits and the bits emitted by the neural network.

## License

This repository is licensed under the Creative Commons Attribution 4.0 International License. See [LICENSE](LICENSE) for the full text.
