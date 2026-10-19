//! Concurrency benchmark: N pipeline handles (each `clone_shared`, own CUDA stream),
//! each diarizing a different file simultaneously. Mirrors a serving workload where
//! N requests are in flight at once.
//!
//! Usage: cargo run --release --features cuda --example cuda_concurrent -- <n> <wav1> [wav2...]
//! SPEAKRS_SMOKE_MODE=cuda|cuda-fast selects the tier (default cuda).

use std::sync::Arc;
use std::time::Instant;

use speakrs::pipeline::PipelineBuilder;
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
use support::ExampleResult;

mod support;

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: cuda_concurrent <n-workers> <wav1> [wav2 ...]  (files cycled if fewer than n)"
        );
        std::process::exit(1);
    }
    let n: usize = args[1].parse()?;
    let files: Vec<String> = args[2..].to_vec();
    let mode = match std::env::var("SPEAKRS_SMOKE_MODE").as_deref() {
        Ok("cuda-fast") => ExecutionMode::CudaFast,
        _ => ExecutionMode::Cuda,
    };
    println!("mode: {mode:?} | workers: {n} | files: {}", files.len());

    let t0 = Instant::now();
    let base = PipelineBuilder::from_pretrained(mode)?.build()?;
    let mut handles = vec![base];
    for _ in 1..n {
        handles.push(handles[0].clone_shared()?);
    }
    let setup_s = t0.elapsed().as_secs_f64();
    println!("setup (1 base + {n} clones): {setup_s:.3}s");

    // Assign each worker a file (cycled) and run all concurrently.
    let mut workers = Vec::new();
    for (i, pipeline) in handles.into_iter().enumerate() {
        let path = files[i % files.len()].clone();
        workers.push(std::thread::spawn(move || -> ExampleResult<(f64, usize, usize)> {
            let audio = support::load_wav_samples(std::path::Path::new(&path))?;
            let duration_s = audio.len() as f64 / 16_000.0;
            let mut pipeline: OwnedDiarizationPipeline = pipeline;
            let t = Instant::now();
            let result = pipeline.run(&audio)?;
            let elapsed = t.elapsed().as_secs_f64();
            Ok((
                duration_s / elapsed,
                result.segments.len(),
                result
                    .segments
                    .iter()
                    .map(|s| s.speaker.clone())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
            ))
        }));
    }

    let t = Instant::now();
    let mut total_rtf = 0.0;
    let mut total_segs = 0;
    let mut i = 0;
    for w in workers {
        let (rtf, segs, speakers) = w.join().map_err(|_| "worker panicked")??;
        i += 1;
        total_rtf += rtf;
        total_segs += segs;
        println!(
            "worker {i:2}: file={:24.24} rtf={rtf:6.0}x segs={segs:4} speakers={speakers}",
            files[(i - 1) % files.len()],
        );
    }
    let wall = t.elapsed().as_secs_f64();
    println!(
        "\nwall: {wall:.3}s | mean per-worker rtf: {:.0}x | aggregate throughput: {:.0}x audio-realtime",
        total_rtf / n as f64,
        n as f64 * total_rtf / n as f64,
    );
    println!("total segments produced: {total_segs}");
    let _ = Arc::new(()); // keep Arc import if unused paths change
    Ok(())
}
