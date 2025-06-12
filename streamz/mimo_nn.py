import asyncio
import numpy as np

class MIMOStream:
    """Simulated MIMO stream that yields input bits and accepts output bits."""

    def __init__(self, num_inputs: int = 2, num_outputs: int = 2, delay: float = 0.01):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.delay = delay

    async def get_input_bits(self) -> np.ndarray:
        """Simulate receiving a vector of input bits from the user/network."""
        await asyncio.sleep(self.delay)
        return np.random.randint(0, 2, self.num_inputs, dtype=np.int32)

    async def send_output_bits(self, bits: np.ndarray) -> None:
        """Simulate sending a vector of output bits back to the user/network."""
        await asyncio.sleep(self.delay)
        print("Sent bits:", bits)


class SimpleNeuralNet:
    """Minimal feed-forward neural network operating on bit vectors."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 8):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, bits: np.ndarray) -> np.ndarray:
        x = bits.astype(np.float32)
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return (out > 0).astype(np.int32)


async def run_stream(stream: MIMOStream, model: SimpleNeuralNet, iterations: int = 10) -> None:
    """Run a simulated real-time loop passing bits through the network."""
    for _ in range(iterations):
        bits = await stream.get_input_bits()
        print("Received bits:", bits)
        out_bits = model.forward(bits)
        await stream.send_output_bits(out_bits)


if __name__ == "__main__":
    stream = MIMOStream(num_inputs=4, num_outputs=2)
    model = SimpleNeuralNet(input_size=4, output_size=2)
    asyncio.run(run_stream(stream, model))
