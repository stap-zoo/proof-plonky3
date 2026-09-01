#!/usr/bin/env python3
"""Run the ZK headline benchmarks and print the paper's LaTeX table.

The Rust command remains the only measurement path.  This script selects its
checked ``--headline`` plan, preserves the raw CSV optionally, and renders the
largest trace height from the sweep into a full ``table`` environment.
Reported costs are:

* ``cells``: all prover-chosen committed cells, amortized per permutation call;
  LogUp extension cells are converted to base-field equivalents (degree two
  for Goldilocks, degree four for BabyBear and KoalaBear).
* ``prove``: trace/witness generation plus ``prove``, totalled per proof
  (not amortized per call — the number of calls a proof batches is reported
  in the caption instead, since it depends on each construction's
  vectorization width).
* ``verify``: verification of the one packed proof, totalled per proof.

Both timing columns share the ``ms/proof`` unit; splitting cost between
generation/proving and verifying is the only distinction left between them.

Tip5 and Monolith each register two arithmetizations of the same permutation.
The checked ``--headline`` plan selects only their LogUp AIRs, so the
lookup-free alternatives are not measured merely to be discarded here.

The shared ``alpha`` column uses KoalaBear/BabyBear/Goldilocks order when
field-specific instances differ. Missing construction/field cells are printed
as ``--`` and are never invented; so is the field width shown in the header,
which is read off the data rather than hardcoded.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from collections import Counter
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


FIELDS = ("koalabear", "babybear", "goldilocks")
FIELD_DISPLAY = {"koalabear": "KoalaBear", "babybear": "BabyBear", "goldilocks": "Goldilocks"}
EXTENSION_DEGREE = {"koalabear": 4, "babybear": 4, "goldilocks": 2}

# These constructions must arrive from the benchmark as lookup jobs. Rejecting
# any ordinary row here verifies that selection happened before measurement.
LOOKUP_CONSTRUCTIONS = {"tip5", "monolith"}

# Literal row groups of the paper table. Keeping this separate from measured
# cost makes the grouping and its midrules stable across benchmark machines.
ROW_GROUPS = (
    ("gmimc", "gmimc2", "neptune", "poseidon1", "poseidon2", "psquarehash"),
    ("anemoi", "griffin", "rescue-prime", "xhash8"),
    ("monolith", "tip5"),
)

# These designs do not carry a monomial S-box exponent in the table.
NO_ALPHA = {"monolith", "psquarehash"}

# A missing field occupies all three cost columns. The first two field groups
# retain the table's vertical separator; the final group does not have one.
UNSUPPORTED_FIELD = {
    "koalabear": r"\multicolumn{3}{c|}{--}",
    "babybear": r"\multicolumn{3}{c|}{--}",
    "goldilocks": r"\multicolumn{3}{c}{--}",
}

# The LaTeX macro each construction's rows are printed under. Anything
# measured but missing here is a table bug, not a name to fall back on, so
# rendering fails loudly instead of printing a raw construction string.
CONSTRUCTION_LATEX = {
    "neptune": r"\neptune",
    "tip5": r"\tipfourprime",
    "monolith": r"\monolith",
    "rescue-prime": r"\rescueprime",
    "gmimc": r"\gmimchash",
    "poseidon1": r"\poseidon",
    "poseidon2": r"\poseidontwo",
    "anemoi": r"\anemoi",
    "xhash8": r"\xhash",
    "griffin": r"\griffin",
    "psquarehash": r"\psquarehash",
    "gmimc2": r"\gmimchashtwo",
}

REQUIRED_COLUMNS = {
    "construction",
    "instance",
    "variant",
    "kind",
    "field",
    "zk",
    "reading",
    "log_n",
    "num_calls",
    "state_width",
    "sbox_degree",
    "witness_base_cells",
    "committed_extension_cells",
    "generate_ns",
    "prove_ns",
    "verify_ns",
}


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
        "--log-n",
        default="10,12,14",
        help="comma-separated trace-height sweep (default: 10,12,14)",
    )
    return parser.parse_args(argv)


def run_bench(repo: Path, threads: int, log_n: str) -> str:
    if threads < 1:
        raise ValueError("--threads must be positive")
    command = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--bin",
        "bench",
        "--",
        "--headline",
        "--field",
        ",".join(FIELDS),
        "--zk",
        "on",
        "--reading",
        "minimum",
        "--log-n",
        log_n,
    ]
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(threads)
    completed = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
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
        raise ValueError("benchmark CSV contains no rows")
    return provenance, rows


def select_largest_height(rows: Iterable[Mapping[str, str]]) -> dict[tuple[str, str, str], Mapping[str, str]]:
    """Validate the headline identity and retain its largest sweep height."""
    chosen: dict[tuple[str, str, str], Mapping[str, str]] = {}
    identities: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows:
        if row["construction"] in LOOKUP_CONSTRUCTIONS and row["kind"] != "lookup":
            raise ValueError(
                f"{row['construction']} reached the table as a non-lookup job; "
                "the headline plan must exclude it before measurement"
            )
        if row["zk"] != "1" or row["reading"] != "minimum":
            raise ValueError("the LaTeX table accepts only ZK minimum-reading rows")
        field = row["field"]
        if field not in FIELDS:
            raise ValueError(f"unexpected field in headline CSV: {field}")
        key = (row["construction"], row["kind"], field)
        identity = (row["instance"], row["variant"])
        previous_identity = identities.setdefault(key, identity)
        if previous_identity != identity:
            raise ValueError(
                f"multiple headline jobs for {key}: {previous_identity} and {identity}"
            )
        previous = chosen.get(key)
        if previous is None or int(row["log_n"]) > int(previous["log_n"]):
            chosen[key] = row
    return chosen


def field_width(chosen: Mapping[tuple[str, str, str], Mapping[str, str]], field: str) -> str:
    """The headline state width measured for `field`, asserted consistent."""
    widths = {row["state_width"] for (_, _, f), row in chosen.items() if f == field}
    if not widths:
        raise ValueError(f"no headline row measured for field {field}")
    if len(widths) != 1:
        raise ValueError(f"inconsistent headline width for {field}: {sorted(widths)}")
    return next(iter(widths))


def trace_height_note(chosen: Mapping[tuple[str, str, str], Mapping[str, str]]) -> str:
    """The `log_n` the headline rows were measured at, as a LaTeX fragment."""
    log_ns = {int(row["log_n"]) for row in chosen.values()}
    if len(log_ns) == 1:
        return f"$2^{{{log_ns.pop()}}}$"
    return f"between $2^{{{min(log_ns)}}}$ and $2^{{{max(log_ns)}}}$"


def latex_power_of_two(value: int) -> str:
    """A positive power of two in the notation used by the table caption."""
    if value <= 0 or value & (value - 1):
        raise ValueError(f"table caption expects a power-of-two count, got {value}")
    return f"$2^{{{value.bit_length() - 1}}}$"


def call_count_note(chosen: Mapping[tuple[str, str, str], Mapping[str, str]]) -> str:
    """How many calls a proof batches, as a caption fragment.

    Vectorization width is per construction, not per field, so this asserts
    one call count per construction and reports the shared value plus any
    named exceptions rather than a single invented number.
    """
    per_construction: dict[str, set[int]] = {}
    for (construction, _kind, _field), row in chosen.items():
        per_construction.setdefault(construction, set()).add(int(row["num_calls"]))
    counts: dict[str, int] = {}
    for construction, values in per_construction.items():
        if len(values) != 1:
            raise ValueError(f"inconsistent call count across fields for {construction}: {values}")
        counts[construction] = next(iter(values))

    common_count, _ = Counter(counts.values()).most_common(1)[0]
    exceptions: dict[int, list[str]] = {}
    for construction, count in counts.items():
        if count != common_count:
            exceptions.setdefault(count, []).append(latex_name(construction))

    note = (
        "Every construction batches "
        f"{latex_power_of_two(common_count)} permutation calls per proof"
    )
    if exceptions:
        parts = [
            f"{' and '.join(sorted(names))} {latex_power_of_two(count)}"
            for count, names in sorted(exceptions.items())
        ]
        note += f", except {'; '.join(parts)}"
    return note


def latex_name(construction: str) -> str:
    try:
        return CONSTRUCTION_LATEX[construction]
    except KeyError as exc:
        raise ValueError(
            f"no LaTeX macro registered for construction {construction!r}; "
            "add it to CONSTRUCTION_LATEX"
        ) from exc


def shared_parameter(cells: Mapping[str, Mapping[str, str]], column: str) -> str:
    values = [cells[field][column] for field in FIELDS if field in cells]
    if not values:
        return "--"
    if all(value == values[0] for value in values):
        return values[0]
    return "/".join(cells[field][column] if field in cells else "--" for field in FIELDS)


def format_cells(value: float) -> str:
    rounded = round(value)
    return str(rounded) if abs(value - rounded) < 1e-9 else f"{value:.1f}"


def format_alpha(construction: str, cells: Mapping[str, Mapping[str, str]]) -> str:
    """The shared $\\alpha$ column, or `--` where it is not applicable."""
    if construction in NO_ALPHA:
        return "--"
    alpha = shared_parameter(cells, "sbox_degree")
    return "--" if alpha == "0" else alpha


def render_rows(
    chosen: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> list[str]:
    """The `tabular` body lines for one already-selected headline set."""
    grouped: dict[str, dict[str, Mapping[str, str]]] = {}
    for (construction, _kind, field), row in chosen.items():
        grouped.setdefault(construction, {})[field] = row

    expected = {construction for group in ROW_GROUPS for construction in group}
    actual = set(grouped)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        unexpected = ", ".join(sorted(actual - expected)) or "none"
        raise ValueError(
            f"headline construction set changed; missing: {missing}; "
            f"unexpected: {unexpected}"
        )

    def render_construction(construction: str) -> str:
        field_rows = grouped[construction]
        values: dict[str, tuple[str, float, float]] = {}
        for field, row in field_rows.items():
            calls = int(row["num_calls"])
            if calls <= 0:
                raise ValueError(f"non-positive call count for {construction}/{field}")
            witness_cells = int(row["witness_base_cells"])
            extension_cells = int(row["committed_extension_cells"])
            cells_per_call = (
                witness_cells + EXTENSION_DEGREE[field] * extension_cells
            ) / calls
            prove_ms_per_proof = (int(row["generate_ns"]) + int(row["prove_ns"])) / 1_000_000
            verify_ms_per_proof = int(row["verify_ns"]) / 1_000_000
            values[field] = (
                format_cells(cells_per_call),
                prove_ms_per_proof,
                verify_ms_per_proof,
            )

        columns = [latex_name(construction), format_alpha(construction, field_rows)]
        for field in FIELDS:
            if field not in values:
                columns.append(UNSUPPORTED_FIELD[field])
            else:
                cells, prove_ms, verify_ms = values[field]
                columns.extend([cells, f"{prove_ms:.2f}", f"{verify_ms:.2f}"])
        return "        " + " & ".join(columns) + r" \\"

    rendered = []
    for index, group in enumerate(ROW_GROUPS):
        rendered.extend(render_construction(construction) for construction in group)
        if index + 1 < len(ROW_GROUPS):
            rendered.extend(["", r"        \midrule", ""])
    return rendered


def render_table(rows: Iterable[Mapping[str, str]], provenance: Iterable[str] = ()) -> str:
    chosen = select_largest_height(rows)
    body = render_rows(chosen)
    calls_note = call_count_note(chosen)
    widths = {field: field_width(chosen, field) for field in FIELDS}

    caption = (
        f"Cost of proving {trace_height_note(chosen)} trace rows. Cells are "
        "prover-committed cells per call, prove is witness generation and "
        f"proving, and verify is verification. {calls_note}."
    )

    comments = [f"% {line}" for line in provenance]
    comments.extend(
        [
            "% alpha uses KoalaBear/BabyBear/Goldilocks order when it differs.",
            "% cells are amortized per call; prove and verify are totalled per proof.",
        ]
    )

    lines = [
        *comments,
        r"\begin{table}[htb!]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \label{tab:plonky-times}",
        r"    \resizebox{\linewidth}{!}{",
        r"    \begin{tabular}{lc|rrr|rrr|rrr}",
        r"        \toprule",
        "        & \\multicolumn{1}{c}{} & "
        f"\\multicolumn{{3}}{{c}}{{{FIELD_DISPLAY['koalabear']} $(t={widths['koalabear']})$}} & "
        f"\\multicolumn{{3}}{{c}}{{{FIELD_DISPLAY['babybear']} $(t={widths['babybear']})$}} & "
        f"\\multicolumn{{3}}{{c}}{{{FIELD_DISPLAY['goldilocks']} $(t={widths['goldilocks']})$}} "
        r"\\ \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
        r"        & \multicolumn{1}{c}{} & cells & prove & \multicolumn{1}{c}{verify} & cells & prove & \multicolumn{1}{c}{verify} & cells & prove & \multicolumn{1}{c}{verify} \\",
        r"        construction & \multicolumn{1}{c}{$\alpha$} & & ms & \multicolumn{1}{c}{ms} & & ms & \multicolumn{1}{c}{ms} & & ms & ms \\",
        r"        \midrule",
        r"        \midrule",
        "",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.input:
        raw = args.input.read_text(encoding="utf-8")
    else:
        raw = run_bench(repo, args.threads, args.log_n)
    if args.raw_output:
        args.raw_output.write_text(raw, encoding="utf-8")
    provenance, rows = parse_csv(raw)
    sys.stdout.write(render_table(rows, provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
