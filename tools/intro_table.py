#!/usr/bin/env python3
"""Run the introduction benchmark and print the paper's LaTeX table.

The table behind Section "Traditional Cryptographic Hash Functions are Not
ZK-friendly": what it costs to prove a fixed amount of *hashing* with the
traditional, Boolean-domain designs Plonky3 arithmetizes, against one
arithmetization-oriented design proved by the same prover in the same
configuration.

The Rust command remains the only measurement path (POLICY §11).  This script
selects its ``--intro`` plan, optionally preserves the raw CSV, and renders a
full ``table`` environment.  Nothing here measures anything, and nothing here
carries a sponge rate of its own: the rates come out of the run's own
provenance block.

The metric: proving per KiB, verifying per proof
-----------------------------------------------
An equal-*calls* comparison is not an equal-*data* one, because one call is a
different amount of hashing per design: a 64-byte compression for BLAKE3 and
SHA-256, a permutation whose sponge rate is 136 bytes for Keccak-f, and 16
field elements for Poseidon2.  So ``--bytes`` sizes each trace by the message
bytes it covers.

The two columns are then reported in *different* units, because the two costs
scale differently, and this is measured rather than assumed:

* **Proving is proportional to the message.**  Sweeping Poseidon2 from 1024 to
  131072 calls -- a 128-fold increase -- moves its cost per KiB by under 5%
  (1.27 to 1.19, non-monotonically); BLAKE3 over a 4-fold increase moves by
  under 1%.  Per KiB is therefore a real rate, and the unequal batches below
  cost the comparison very little.
* **Verifying is near-constant per proof.**  Over that same 128-fold increase
  Poseidon2's verification grows only from 12.3 ms to 17.8 ms, and BLAKE3's
  from 42.1 to 44.9 over 4x.  A per-KiB verify figure would therefore be
  ``constant / batch size`` -- it would report how large each batch happened to
  be rather than anything about the design, and with batches differing by 41%
  here it would misattribute that spread.  So verification is reported **per
  proof**, with the batch each proof covered stated in the caption.

Why the batches are still not exactly equal
-------------------------------------------
The reachable batch sizes are quantised and they double, so they cannot be made
to coincide:

* ``p3-blake3-air`` and ``p3-sha256-air`` require a power-of-two row count, so
  their batch is exactly ``64 * 2^k`` bytes.
* ``p3-poseidon2-air`` packs 8 calls to a row, so its batch is ``48 * 2^k``.
* ``p3-keccak-air`` spends 24 rows on a call, so its batch is
  ``136 * floor(2^k / 24)`` -- which lands at ``0.7076`` times the nearest
  power-of-two-block anchor at *every* height.  That ratio is invariant, so no
  choice of target improves it: Keccak-f is either ~29% under or ~41% over.

Under is the choice made here.  Never exceeding the target means no row gets a
bigger batch to amortise its fixed costs over, so a design that cannot land on
the target is charged for the shortfall rather than credited for an overshoot.
The per-KiB metric then normalises what remains, and the caption reports the
batch each row actually proved.

Why the target defaults to 64 KiB
---------------------------------
It is the anchor that puts three of the four rows within 3% of each other while
staying inside the 2^14 trace-height ceiling that KoalaBear's 100-bit proven
security imposes (``harness::config``).  At 64 KiB the heights are 2^10, 2^10,
2^13 and 2^7, and the only row that pads is Keccak-f, by 8 rows of 8192.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


# One field: the point is the Boolean/algebraic gap, not the choice of prime.
# KoalaBear because it is the fastest 31-bit prime in the paper's main table,
# so the arithmetization-oriented side is shown at its best.
FIELD = "koalabear"

# Message bytes each proof must not exceed. See the module docstring.
DEFAULT_BYTES = 64 * 1024

# How each measured construction is printed. `\poseidontwo` is the macro the
# main table already uses, so the two tables cannot print one instance under two
# names. The traditional designs have no macro in the paper and are spelled out.
#
# There are deliberately no byte counts here: the rates come from the run's
# provenance block, which is where the measurement itself reported them.
CONSTRUCTION_LATEX = {
    "blake3": r"\textsc{Blake3}",
    "sha256": r"\textsc{Sha}-256",
    "keccak-f": r"\textsc{Keccak}-$f$[1600]",
    "poseidon2": r"\poseidontwo",
}

# Which rows are the Boolean-domain baseline. Ordering and caption only.
TRADITIONAL = ("blake3", "sha256", "keccak-f")

REQUIRED_COLUMNS = {
    "construction",
    "instance",
    "variant",
    "field",
    "zk",
    "reading",
    "log_n",
    "num_calls",
    "calls_per_row",
    "rows_per_call",
    "max_constraint_degree",
    "log_blowup",
    "security_bits",
    "generate_ns",
    "prove_ns",
    "verify_ns",
}

RATE_PREFIX = "absorbed-bytes-per-call:"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="render an existing benchmark CSV instead of running Cargo",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="save Cargo's raw CSV/provenance stream at this path",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("RAYON_NUM_THREADS", "16")),
        help="Rayon worker count (default: RAYON_NUM_THREADS or 16)",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=DEFAULT_BYTES,
        help=f"message bytes each proof must not exceed (default: {DEFAULT_BYTES})",
    )
    return parser.parse_args(argv)


def run_bench(repo: Path, threads: int, target_bytes: int) -> str:
    if threads < 1:
        raise ValueError("--threads must be positive")
    if target_bytes < 1:
        raise ValueError("--bytes must be positive")
    command = [
        "cargo", "run", "--release", "--quiet", "--bin", "bench", "--",
        "--intro",
        "--field", FIELD,
        "--zk", "on",
        "--reading", "minimum",
        "--bytes", str(target_bytes),
    ]
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(threads)
    completed = subprocess.run(
        command, cwd=repo, env=environment, check=True, text=True, stdout=subprocess.PIPE
    )
    return completed.stdout


def parse_csv(raw: str) -> tuple[list[str], list[dict[str, str]]]:
    provenance = [line[1:].strip() for line in raw.splitlines() if line.startswith("#")]
    csv_text = "\n".join(
        line for line in raw.splitlines() if line.strip() and not line.startswith("#")
    )
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise ValueError("benchmark output contains no CSV header")
    missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
    if missing:
        raise ValueError(f"benchmark CSV is missing columns: {', '.join(sorted(missing))}")
    rows = list(reader)
    if not rows:
        raise ValueError(
            "benchmark CSV contains no rows; a skipped job means the requested "
            "--bytes needs a trace height the 100-bit target does not admit"
        )
    return provenance, rows


def parse_rates(provenance: Iterable[str]) -> dict[str, int]:
    """The sponge rates the run itself reported.

    Read back rather than restated, so this script cannot disagree with the
    measurement about how much message a call carried (POLICY §8: the rate is
    reported, never chosen).
    """
    for line in provenance:
        if line.startswith(RATE_PREFIX):
            body = line[len(RATE_PREFIX):].strip()
            rates = {}
            for entry in body.split(","):
                name, _, value = entry.partition("=")
                rates[name.strip()] = int(value)
            return rates
    raise ValueError(
        "the benchmark output carries no `absorbed-bytes-per-call` line; "
        "the run was not made with --bytes, so its batches are not comparable by data"
    )


def select(rows: Iterable[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    """One row per construction, with the table's identity assumptions asserted."""
    chosen: dict[str, Mapping[str, str]] = {}
    for row in rows:
        if row["zk"] != "1" or row["reading"] != "minimum":
            raise ValueError("the LaTeX table accepts only ZK minimum-reading rows")
        if row["field"] != FIELD:
            raise ValueError(f"unexpected field in intro CSV: {row['field']}")
        construction = row["construction"]
        if construction in chosen:
            raise ValueError(f"more than one intro row for {construction}")
        chosen[construction] = row
    unknown = set(chosen).difference(CONSTRUCTION_LATEX)
    if unknown:
        raise ValueError(
            f"no LaTeX name registered for {', '.join(sorted(unknown))}; "
            "add it to CONSTRUCTION_LATEX"
        )
    missing = set(CONSTRUCTION_LATEX).difference(chosen)
    if missing:
        raise ValueError(f"the intro plan measured no row for {', '.join(sorted(missing))}")
    return chosen


def shared(chosen: Mapping[str, Mapping[str, str]], column: str) -> str:
    """A column every row must agree on, which is why the table is like-for-like."""
    values = {row[column] for row in chosen.values()}
    if len(values) != 1:
        raise ValueError(f"rows disagree on {column}: {sorted(values)}")
    return next(iter(values))


def kib(byte_count: int) -> float:
    return byte_count / 1024


def latex_number(value: int) -> str:
    return f"{value:,}".replace(",", r"\,")


def batch_note(chosen: Mapping[str, Mapping[str, str]], rates: Mapping[str, int]) -> str:
    """What each proof actually covered, per row.

    Reported rather than intended: the reachable batch sizes are quantised, so
    a row's batch is what its layout could reach under the target, not the
    target.
    """
    parts = []
    ordered = sorted(chosen.items(), key=lambda item: (item[0] not in TRADITIONAL, item[0]))
    for name, row in ordered:
        calls = int(row["num_calls"])
        data = calls * rates[name]
        parts.append(
            f"{CONSTRUCTION_LATEX[name]} {latex_number(calls)} calls "
            f"({kib(data):.1f}\\,KiB)"
        )
    return "; ".join(parts)


def rows_occupied(row: Mapping[str, str]) -> int:
    """Trace rows a row's batch occupies.

    Both factors are needed and forgetting either gives nonsense: calls are
    packed `calls_per_row` to a row *and* spread over `rows_per_call` rows.
    Poseidon2 is the case that catches it -- 1024 calls at 8 to a row occupy
    128 rows, not 1024.
    """
    calls = int(row["num_calls"])
    return calls * int(row["rows_per_call"]) // int(row["calls_per_row"])


def padding_note(chosen: Mapping[str, Mapping[str, str]]) -> str:
    """Which rows pad, if any, with the exact cost."""
    parts = []
    for name, row in chosen.items():
        height = 1 << int(row["log_n"])
        used = rows_occupied(row)
        if used > height:
            raise ValueError(
                f"{name} occupies {used} rows in a {height}-row trace; "
                "the layout labels and the measured height disagree"
            )
        if used != height:
            parts.append(
                f"{CONSTRUCTION_LATEX[name]}'s {row['rows_per_call']}-row layout leaves "
                f"{latex_number(height - used)} of {latex_number(height)} rows padding"
            )
    if not parts:
        return "Every trace is full: no row pads."
    prefix = "Only " if len(parts) == 1 else ""
    return prefix + "; ".join(parts) + "."


def render_rows(
    chosen: Mapping[str, Mapping[str, str]], rates: Mapping[str, int]
) -> list[str]:
    """The `tabular` body: cost per KiB, traditional designs above the AO one."""
    rendered = []
    for name, row in chosen.items():
        data_kib = kib(int(row["num_calls"]) * rates[name])
        # Proving scales with the message, so it is a rate. Verification does
        # not, so dividing it by the batch would report the batch.
        prove = (int(row["generate_ns"]) + int(row["prove_ns"])) / 1_000_000 / data_kib
        verify = int(row["verify_ns"]) / 1_000_000
        line = (
            f"        {CONSTRUCTION_LATEX[name]} & "
            f"{prove:,.1f} & {verify:,.1f}".replace(",", r"\,") + r" \\"
        )
        rendered.append((name in TRADITIONAL, prove, line))
    rendered.sort(key=lambda item: (not item[0], -item[1]))
    lines = [line for _, _, line in rendered]
    boundary = sum(1 for is_traditional, _, _ in rendered if is_traditional)
    if 0 < boundary < len(lines):
        lines.insert(boundary, r"        \midrule")
    return lines


def render_table(rows: Iterable[Mapping[str, str]], provenance: Iterable[str] = ()) -> str:
    provenance = list(provenance)
    rates = parse_rates(provenance)
    chosen = select(rows)
    missing_rates = set(chosen).difference(rates)
    if missing_rates:
        raise ValueError(f"the run reported no rate for {', '.join(sorted(missing_rates))}")

    body = render_rows(chosen, rates)
    log_blowup = int(shared(chosen, "log_blowup"))
    degree = shared(chosen, "max_constraint_degree")
    security = min(int(row["security_bits"]) for row in chosen.values())

    caption = (
        "Cost of proving a fixed amount of \\emph{hashing} in Plonky3, over "
        "\\textsc{KoalaBear} with zero knowledge. \\emph{prove} is witness "
        "generation plus proving, divided by the message that proof covered; "
        "\\emph{verify} is verification of the same proof, reported per proof "
        "rather than per KiB. The two units differ because the two costs scale "
        "differently, which is measured rather than assumed: proving is "
        "proportional to the message, its per-KiB figure moving by under 5% as "
        "the batch grows 128-fold, while verification is near-constant per "
        "proof over that same range, so a per-KiB verify figure would report "
        "the batch size rather than the design. Costing by message rather than "
        "by call matters because one call is a different amount of hashing per "
        "design -- a 64-byte compression for \\textsc{Blake3} and "
        "\\textsc{Sha}-256, a permutation of sponge rate 136 bytes for "
        "\\textsc{Keccak}-$f$[1600], and 16 \\textsc{KoalaBear} elements "
        "(48 bytes at three bytes per element) for \\poseidontwo{} at $t=24$, "
        f"whose rate is the reference's. The proofs cover {batch_note(chosen, rates)}; "
        "the batches differ because each layout's reachable sizes are quantised "
        "and double, so they cannot be made to coincide, and each row takes the "
        f"largest batch that does not exceed 64\\,KiB. {padding_note(chosen)} "
        f"All four arithmetizations have maximum constraint degree {degree}, so "
        f"every row is proved at the same FRI blowup $2^{{{log_blowup}}}$ and at "
        f"$\\geq {security}$ bits of proven security; the rows therefore differ "
        "only in the arithmetization. The instances are Plonky3's own, and "
        "\\poseidontwo{} is the same instance as \\Cref{tab:plonky-times}."
    )

    comments = [f"% {line}" for line in provenance]
    comments.append("% generated by tools/intro_table.py; do not edit by hand.")
    comments.append("% prove = generate + prove; both columns are per KiB of message.")

    lines = [
        *comments,
        r"\begin{table}[htb!]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \label{tab:intro-zk-unfriendly}",
        r"    \begin{tabular}{lrr}",
        r"        \toprule",
        r"        construction & prove & verify \\",
        r"                     & (ms/KiB) & (ms/proof) \\",
        r"        \midrule",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.input:
        raw = args.input.read_text(encoding="utf-8")
    else:
        raw = run_bench(repo, args.threads, args.bytes)
    if args.raw_output:
        args.raw_output.write_text(raw, encoding="utf-8")
    provenance, rows = parse_csv(raw)
    sys.stdout.write(render_table(rows, provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
