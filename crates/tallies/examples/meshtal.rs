//! Index a meshtal file and read it, timing each step and reporting peak
//! memory: `cargo run --release --example meshtal -- <file> [tally]
//! [--input <mcnp input>] [--all]`. With a tally, one bin of it and then
//! the whole of it are read; with `--all`, every tally, each with a
//! checksum of its values (to compare two builds bit for bit).

use std::time::Instant;

use aci_tallies::common::Input;
use aci_tallies::meshtal::Reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or("give a meshtal file")?;
    let input = args
        .iter()
        .position(|a| a == "--input")
        .and_then(|k| args.get(k + 1));
    let tally: Option<u32> = args.get(1).and_then(|a| a.parse().ok());

    let t = Instant::now();
    let reader = match input {
        Some(i) => {
            let input = Input::read(i)?;
            println!("input read in {:.2} s", t.elapsed().as_secs_f64());
            Reader::open_with_input(path, &input)?
        }
        None => Reader::open(path)?,
    };
    println!("indexed in {:.2} s", t.elapsed().as_secs_f64());
    for info in reader.tallies() {
        println!(
            "  tally {}: {:?}, out={}",
            info.id,
            info.shape(),
            info.out_format.name()
        );
    }
    for (id, why) in reader.failed() {
        println!("  tally {id} can't be read: {why}");
    }

    if args.iter().any(|a| a == "--all") {
        let t = Instant::now();
        let all = reader.read_all()?;
        println!("read all in {:.2} s", t.elapsed().as_secs_f64());
        let hash = |a: &mut dyn Iterator<Item = &f64>| {
            a.fold(0u64, |h, v| h.wrapping_mul(31).wrapping_add(v.to_bits()))
        };
        for t in &all {
            println!(
                "{} {:?} {:016x} {:016x} {} {} {}",
                t.id,
                t.result.shape(),
                hash(&mut t.result.iter()),
                hash(&mut t.rel_error.iter()),
                t.volume.as_ref().map_or(0, |v| hash(&mut v.iter())),
                t.result.iter().filter(|v| v.is_finite()).count(),
                t.warnings.len()
            );
        }
    } else if let Some(id) = tally.or_else(|| reader.tallies().next().map(|t| t.id)) {
        let info = reader.tally(id)?;
        let (e, k) = (
            info.energy.total_index().unwrap_or(0),
            info.time.total_index().unwrap_or(0),
        );
        let t = Instant::now();
        let bin = reader.read_bin(id, e, k)?;
        println!(
            "tally {id} bin ({e}, {k}) read in {:.2} s ({} warnings)",
            t.elapsed().as_secs_f64(),
            bin.warnings.len()
        );
        let t = Instant::now();
        let whole = reader.read(id)?;
        println!(
            "tally {id} read whole in {:.2} s, shape {:?}",
            t.elapsed().as_secs_f64(),
            whole.result.shape()
        );
    }
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(peak) = status.lines().find(|l| l.starts_with("VmHWM")) {
            println!("{peak}");
        }
    }
    Ok(())
}
