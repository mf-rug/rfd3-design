#!/usr/bin/env python3
"""Fold a single protein sequence with Meta's hosted ESMFold endpoint.

Usage:
    python esmfold.py <sequence | fasta_path> [-o out.pdb] [--cif]

The endpoint is public, no auth, typically a few seconds for sequences up
to ~400 residues. Output is a PDB (or mmCIF with --cif); the B-factor
column carries per-residue pLDDT as a **fraction (0-1)**. Mean pLDDT
printed to stderr is rescaled to the conventional 0-100 scale.

Limits: occasional rate-limiting / downtime; sequences much longer than
~400 aa often fail with 400/timeout; monomer only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

API = "https://api.esmatlas.com/foldSequence/v1"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_sequence(arg: str) -> tuple[str, str]:
    """Return (name, sequence). `arg` is either a literal sequence or a
    path to a FASTA file (single record)."""
    p = Path(arg)
    if p.is_file():
        lines = p.read_text().splitlines()
        name = lines[0].lstrip(">").split()[0] if lines and lines[0].startswith(">") else p.stem
        seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    else:
        name, seq = "input", arg
    seq = seq.strip().replace(" ", "").replace("\n", "").upper()
    bad = set(seq) - VALID_AA
    if bad:
        raise ValueError(f"Sequence contains non-standard residues: {sorted(bad)}")
    return name, seq


def fold(seq: str, fmt: str = "pdb", timeout: int = 120) -> str:
    r = requests.post(
        f"{API}/{fmt}",
        data=seq,
        headers={"Content-Type": "text/plain"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


def mean_plddt_from_pdb(pdb_text: str) -> float | None:
    """Extract mean per-CA pLDDT (B-factor column), rescaled to 0-100.

    The API stores pLDDT as a 0-1 fraction; if max value <= 1 we scale
    by 100 to return the conventional 0-100 scale.
    """
    vals = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return mean * 100 if max(vals) <= 1.0 else mean


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="Amino-acid sequence, or path to a FASTA file")
    ap.add_argument("-o", "--out", help="Output path (default: <name>.pdb in cwd)")
    ap.add_argument("--cif", action="store_true", help="Request mmCIF instead of PDB")
    args = ap.parse_args()

    name, seq = read_sequence(args.sequence)
    fmt = "cif" if args.cif else "pdb"
    out = Path(args.out) if args.out else Path(f"{name}.{fmt}")

    print(f"folding {name} ({len(seq)} aa) → {out}", file=sys.stderr)
    text = fold(seq, fmt=fmt)
    out.write_text(text)

    if not args.cif:
        plddt = mean_plddt_from_pdb(text)
        if plddt is not None:
            print(f"mean pLDDT: {plddt:.1f}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
