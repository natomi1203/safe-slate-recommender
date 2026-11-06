use std::env;
use std::fs;
use std::net::SocketAddr;

use _core::sidecar::RerankerService;
use _core::sidecar::proto::safe_slate_reranker_server::SafeSlateRerankerServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listen_addr: SocketAddr = env::var("SAFE_SLATE_LISTEN_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50051".to_owned())
        .parse()?;
    if !listen_addr.ip().is_loopback() {
        return Err("SAFE_SLATE_LISTEN_ADDR must be loopback".into());
    }
    let contract_digest = env::var("SAFE_SLATE_CONTRACT_SHA256")?;
    let model_path = env::var("SAFE_SLATE_MODEL_PATH")?;
    let model_bytes = fs::read(model_path)?;
    let service = RerankerService::from_model_bytes(contract_digest, &model_bytes)?;

    Server::builder()
        .add_service(SafeSlateRerankerServer::new(service))
        .serve_with_shutdown(listen_addr, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
