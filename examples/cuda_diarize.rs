//! CUDA smoke test: download community-1 assets (via the pinned avencera/speakrs-models
//! revision), load the native CUDA pipeline, diarize one WAV, and report timings.
//!
//! Usage: cargo run --release --features cuda --example cuda_diarize -- <audio.wav> [file-id]
//!
//! Modes: set SPEAKRS_SMOKE_MODE=cuda|cuda-fast to pick the tier (default cuda).

mod support;

use std::path::Path;
use std::time::Instant;

use speakrs::pipeline::PipelineBuilder;
use speakrs::ExecutionMode;
use support::ExampleResult;

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "Usage: cargo run --release --features cuda --example cuda_diarize -- <audio.wav> [file-id]"
        );
        std::process::exit(1);
    }

    let audio_path = Path::new(&args[1]);
    let file_id = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| support::file_id_from_path(audio_path));

    let mode = match std::env::var("SPEAKRS_SMOKE_MODE").as_deref() {
        Ok("cuda-fast") => ExecutionMode::CudaFast,
        _ => ExecutionMode::Cuda,
    };
    println!("mode: {mode:?}");

    let audio = support::load_wav_samples(audio_path)?;
    let duration_s = audio.len() as f64 / 16_000.0;
    println!("audio: {duration_s:.2}s ({} samples)", audio.len());

    // from_pretrained downloads the pinned safetensors/PLDA files into the HF
    // cache on first use (~31 MB total for CUDA modes) and resolves paths.
    let t0 = Instant::now();
    let mut pipeline = PipelineBuilder::from_pretrained(mode)?.build()?;
    let load_s = t0.elapsed().as_secs_f64();

    // Warm up with a second run: the first includes any lazy JIT/context setup.
    let t1 = Instant::now();
    let result = pipeline.run(&audio)?;
    let cold_s = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let result = pipeline.run(&audio)?;
    let warm_s = t2.elapsed().as_secs_f64();
    println!(
        "load: {load_s:.3}s | cold run: {cold_s:.3}s ({:.0}x) | warm run: {warm_s:.3}s ({:.0}x)",
        duration_s / cold_s,
        duration_s / warm_s,
    );

    let mut names: Vec<String> = result
        .segments
        .iter()
        .map(|s| s.speaker.clone())
        .collect();
    names.sort();
    names.dedup();
    let speakers: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    println!(
        "speakers: {} ({}) | segments: {} | exclusive: {}",
        speakers.len(),
        speakers.join(","),
        result.segments.len(),
        result.exclusive_segments.len(),
    );
    print!("{}", result.rttm(&file_id));
    Ok(())
}
