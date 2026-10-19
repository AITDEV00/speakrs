//! Batch throughput benchmark: run a directory of WAVs through the overlapped
//! batch pipeline (`run_batch`), the configuration upstream reports its headline
//! RTFx with. Reports total audio minutes, wall time, aggregate RTFx, and
//! per-file counts. Compare modes with SPEAKRS_SMOKE_MODE.
//!
//! Usage: cargo run --release --features cuda --example cuda_batch -- <dir-or-files...>
//! SPEAKRS_SMOKE_MODE=cuda|cuda-fast (default cuda)

mod support;

use std::path::PathBuf;
use std::time::Instant;

use speakrs::pipeline::{BatchInput, PipelineBuilder};
use speakrs::ExecutionMode;
use support::ExampleResult;

fn main() -> ExampleResult<()> {
    support::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cuda_batch <dir-or-wav...>");
        std::process::exit(1);
    }

    let mode = match std::env::var("SPEAKRS_SMOKE_MODE").as_deref() {
        Ok("cuda-fast") => ExecutionMode::CudaFast,
        _ => ExecutionMode::Cuda,
    };

    // Collect wav files from dirs, or take explicit paths.
    let mut paths: Vec<PathBuf> = Vec::new();
    for a in &args[1..] {
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut wavs: Vec<PathBuf> = std::fs::read_dir(&p)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
                .collect();
            wavs.sort();
            paths.extend(wavs);
        } else {
            paths.push(p);
        }
    }
    if paths.is_empty() {
        return Err("no wav files found".into());
    }

    println!("mode: {mode:?} | files: {}", paths.len());

    // Pre-load all audio in parallel on CPU threads (upstream's bench counts WAV
    // loading in its timer too, but pre-decode isolates the GPU path; we report both).
    let t_load = Instant::now();
    let mut files: Vec<(PathBuf, Vec<f32>)> = Vec::with_capacity(paths.len());
    for p in &paths {
        files.push((p.clone(), support::load_wav_samples(p)?));
    }
    let decode_s = t_load.elapsed().as_secs_f64();
    let total_s: f64 = files.iter().map(|(_, a)| a.len() as f64 / 16_000.0).sum();
    println!(
        "decoded {} files, {:.1} min audio in {decode_s:.3}s",
        files.len(),
        total_s / 60.0
    );

    let t0 = Instant::now();
    let mut pipeline = PipelineBuilder::from_pretrained(mode)?.build()?;
    let load_s = t0.elapsed().as_secs_f64();

    let inputs: Vec<BatchInput<'_>> = files
        .iter()
        .map(|(p, audio)| BatchInput {
            audio,
            file_id: p.file_stem().and_then(|s| s.to_str()).unwrap_or("file"),
        })
        .collect();

    // Warm pass on the first (shortest reasonable) file to absorb JIT/context
    let t1 = Instant::now();
    let results = pipeline.run_batch(&inputs)?;
    let batch_s = t1.elapsed().as_secs_f64();

    // Second pass: fully warm
    let t2 = Instant::now();
    let results = pipeline.run_batch(&inputs)?;
    let warm_s = t2.elapsed().as_secs_f64();

    println!(
        "pipeline load: {load_s:.3}s | cold batch: {batch_s:.3}s ({:.0}x) | warm batch: {warm_s:.3}s ({:.0}x)",
        total_s / batch_s,
        total_s / warm_s,
    );
    println!(
        "throughput: {:.1} min of audio per second",
        total_s / 60.0 / warm_s
    );

    let mut speakers_total = 0usize;
    let mut segs_total = 0usize;
    for (i, r) in results.iter().enumerate() {
        let sp: std::collections::BTreeSet<&str> =
            r.segments.iter().map(|s| s.speaker.as_str()).collect();
        speakers_total += sp.len();
        segs_total += r.segments.len();
        println!(
            "  {:30.30}: {} speakers, {} segments",
            paths[i].file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
            sp.len(),
            r.segments.len(),
        );
    }
    println!(
        "totals: {} speaker-clusters, {} segments",
        speakers_total, segs_total
    );
    let _ = decode_s;
    Ok(())
}
