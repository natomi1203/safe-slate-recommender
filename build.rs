fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    // SAFETY: Cargo runs this build script in its own process, and no threads are
    // started before tonic-build reads PROTOC.
    unsafe { std::env::set_var("PROTOC", protoc) };

    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["proto/safe_slate_reranker.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/safe_slate_reranker.proto");
    Ok(())
}
