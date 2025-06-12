use streamz_rs::{MIMOStream, SimpleNeuralNet, live_stream};

#[tokio::main]
async fn main() {
    let stream = MIMOStream::new(16, 10);
    let mut net = SimpleNeuralNet::new(16, 8, 16);
    if let Err(e) = live_stream(&stream, &mut net).await {
        eprintln!("Error: {}", e);
    }
}
