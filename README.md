# Plonky3 hash benchmarks

This repository contains the implementation and benchmark paths needed to regenerate the LaTeX performance tables.

## Benchmarked constructions

- Anemoi
- GMiMC
- GMiMC2
- Griffin
- Monolith
- Neptune
- Poseidon1
- Poseidon2
- pSquareHash
- Rescue-Prime
- Tip5
- XHash8
- XHash16

The generated headline table selects one benchmarked configuration per supported construction and field. Unsupported cells are rendered as `--`.

## Traditional-hash comparison

The second table compares the Plonky3 AIRs for BLAKE3, SHA-256, and Keccak-f[1600] against the headline Poseidon2 configuration. These traditional hashes are comparison baselines rather than registered constructions in the main table.

All four rows use KoalaBear with ZK enabled and the minimum-blowup reading. The default workload is at most 64 KiB of message data per proof; the generated caption reports the actual batch size reached by each AIR layout.

## Prerequisites

- A recent stable Rust toolchain with Cargo and Rust 2024 edition support (Rust 1.85 or newer).
- Git and network access for Cargo's first fetch of the pinned Plonky3 revision.
- Python 3.10 or newer. The table script uses only the Python standard library.
- A Unix-like shell. Benchmark results depend on the machine because the workspace builds with `-Ctarget-cpu=native`.

The benchmark defaults to 16 Rayon worker threads. Use the same thread count when comparing runs across machines.

## Generate the LaTeX tables

### Main construction table

From the repository root, run:

```sh
python3 tools/benchmark_table.py --threads 16 --raw-output rows.csv > table.tex
```

This builds and runs the release benchmark, saves the complete provenance and CSV stream in `rows.csv`, and writes the LaTeX table to `table.tex`.

To render the table again without rerunning the benchmarks:

```sh
python3 tools/benchmark_table.py --input rows.csv > table.tex
```

### Traditional-hash comparison

Run:

```sh
python3 tools/intro_table.py --threads 16 --raw-output intro_rows.csv > intro_table.tex
```

To render this table again without rerunning the benchmarks:

```sh
python3 tools/intro_table.py --input intro_rows.csv > intro_table.tex
```
