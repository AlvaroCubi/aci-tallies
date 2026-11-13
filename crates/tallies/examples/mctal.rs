//! Index an MCTAL file and read it, timing each step and reporting peak
//! memory: `cargo run --release --example mctal -- <file> [tally] [--all]`.
//! With `--all`, every tally is read, each with a checksum of its values.

use std::time::Instant;

use aci_tallies::mctal::{Reader, DIMS};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or("give an MCTAL file")?;
    let tally: Option<u32> = args.get(1).and_then(|a| a.parse().ok());

    let t = Instant::now();
    let reader = Reader::open(path)?;
    let h = reader.header();
    println!(
        "indexed in {:.2} s: {} {} {}, {} histories, {} tallies",
        t.elapsed().as_secs_f64(),
        h.code,
        h.version,
        h.probid,
        h.histories,
        h.tallies.len()
    );
    for w in reader.warnings() {
        println!("  warning: {w}");
    }
    for info in reader.tallies() {
        let dims: Vec<String> = DIMS
            .iter()
            .zip(info.shape())
            .filter(|(_, n)| *n > 1)
            .map(|(d, n)| format!("{d}={n}"))
            .collect();
        println!(
            "  tally {}: {} {:?} {:?}{}",
            info.id,
            dims.join(" "),
            info.detector,
            info.particles.iter().collect::<String>(),
            info.comment
                .as_ref()
                .map_or(String::new(), |c| format!(" {c:?}"))
        );
    }
    for (id, why) in reader.failed() {
        println!("  tally {id} can't be read: {why}");
    }
    if let Some(k) = reader.kcode() {
        match k {
            Ok(k) => println!("  kcode: {} cycles, {} inactive", k.cycles, k.inactive),
            Err(e) => println!("  kcode can't be read: {e}"),
        }
    }

    if args.iter().any(|a| a == "--all") {
        let t = Instant::now();
        let all = reader.read_all()?;
        println!("read all in {:.2} s", t.elapsed().as_secs_f64());
        for t in &all {
            let hash = t
                .result
                .iter()
                .chain(&t.rel_error)
                .fold(0u64, |h, v| h.wrapping_mul(31).wrapping_add(v.to_bits()));
            println!(
                "{} {:?} {hash:016x} {}",
                t.id,
                t.result.shape(),
                t.warnings.len()
            );
        }
    } else if let Some(id) = tally.or_else(|| reader.tallies().next().map(|t| t.id)) {
        let t = Instant::now();
        let whole = reader.read(id)?;
        println!(
            "tally {id} read in {:.2} s, shape {:?}, {} warnings",
            t.elapsed().as_secs_f64(),
            whole.result.shape(),
            whole.warnings.len()
        );
    }
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(peak) = status.lines().find(|l| l.starts_with("VmHWM")) {
            println!("{peak}");
        }
    }
    Ok(())
}
