#!/usr/bin/env python3
"""Fold a single protein sequence with ESMFold.

Two backends, picked automatically:

  * **esmatlas** — Meta's public endpoint (no auth, no key). Fast for short
    sequences but caps at ~400 aa (longer → 413/400/timeout). Default for
    sequences ≤ 400 aa.
  * **nvidia** — NVIDIA's hosted ESMFold NIM (GPU, same model), accepts up to
    **1024 aa**. Used automatically as a FALLBACK when the sequence is
    > 400 aa, or when esmatlas errors out. Needs a free API key (see below).

Usage:
    python esmfold.py <sequence | fasta_path> [-o out.pdb] [--cif]
                      [--backend {auto,esmatlas,nvidia}]

NVIDIA key (only needed for the nvidia backend): set NVIDIA_API_KEY (an
'nvapi-...' key from https://build.nvidia.com → any model → "Get API Key"),
or store it in ~/.nvidia/.env as `NVIDIA_API_KEY=nvapi-...`.

B-factor column carries per-residue pLDDT. NOTE the scale differs by backend:
esmatlas writes a **0-1 fraction**, NVIDIA writes the conventional **0-100**.
The mean pLDDT printed to stderr is always rescaled to 0-100, and the written
file's scale is reported so you know what the B-factors mean.

Limits: esmatlas has occasional rate-limiting/downtime; both backends are
monomer-only here; NVIDIA caps at 1024 aa.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import requests

ESMATLAS_API = "https://api.esmatlas.com/foldSequence/v1"
NVIDIA_API = os.environ.get(
    "ESMFOLD_NIM_URL", "https://health.api.nvidia.com/v1/biology/meta/esmfold"
)
ESMATLAS_MAX = 400  # above this, esmatlas reliably fails — go straight to NVIDIA
NVIDIA_MAX = 1024
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


# --- esmatlas (public, no key) ------------------------------------------------

def fold_esmatlas(seq: str, fmt: str = "pdb", timeout: int = 120) -> str:
    r = requests.post(
        f"{ESMATLAS_API}/{fmt}",
        data=seq,
        headers={"Content-Type": "text/plain"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


# --- NVIDIA NIM fallback (key required, up to 1024 aa) ------------------------

def _load_nvidia_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVCF_API_KEY")
    if not key:
        envfile = Path("~/.nvidia/.env").expanduser()
        if envfile.is_file():
            for line in envfile.read_text().splitlines():
                m = re.match(r"\s*(?:NVIDIA_API_KEY|NVCF_API_KEY)\s*=\s*(\S+)", line)
                if m:
                    key = m.group(1).strip().strip('"').strip("'")
                    break
    if not key:
        raise SystemExit(
            "ERROR: NVIDIA backend needs an API key. Set NVIDIA_API_KEY (nvapi-...) "
            "or put it in ~/.nvidia/.env. Get one free at https://build.nvidia.com"
        )
    return key


def fold_nvidia(seq: str, fmt: str = "pdb", timeout: int = 300) -> str:
    if fmt != "pdb":
        print("note: NVIDIA backend returns PDB only; ignoring --cif", file=sys.stderr)
    if len(seq) > NVIDIA_MAX:
        raise SystemExit(f"ERROR: {len(seq)} aa exceeds the NVIDIA NIM {NVIDIA_MAX}-residue cap.")
    key = _load_nvidia_key()
    r = requests.post(
        NVIDIA_API,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"sequence": seq},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise SystemExit(f"ERROR {r.status_code} from {NVIDIA_API}\n{r.text[:1000]}")
    data = r.json()
    if isinstance(data, dict):
        if data.get("pdbs"):
            return data["pdbs"][0]
        if "pdb" in data:
            return data["pdb"]
    raise SystemExit(f"Unexpected NVIDIA response: {str(data)[:500]}")


def fold(seq: str, fmt: str, backend: str) -> tuple[str, str]:
    """Return (pdb/cif text, backend_used) honoring the backend policy.

    auto: esmatlas for ≤400 aa with NVIDIA fallback on error; NVIDIA directly
    for >400 aa.
    """
    if backend == "esmatlas":
        return fold_esmatlas(seq, fmt=fmt), "esmatlas"
    if backend == "nvidia":
        return fold_nvidia(seq, fmt=fmt), "nvidia"

    # auto
    if len(seq) > ESMATLAS_MAX:
        print(f"{len(seq)} aa > {ESMATLAS_MAX}: using NVIDIA NIM backend", file=sys.stderr)
        return fold_nvidia(seq, fmt=fmt), "nvidia"
    try:
        return fold_esmatlas(seq, fmt=fmt), "esmatlas"
    except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
        print(f"esmatlas failed ({e}); falling back to NVIDIA NIM", file=sys.stderr)
        return fold_nvidia(seq, fmt=fmt), "nvidia"


def mean_plddt_from_pdb(pdb_text: str) -> float | None:
    """Extract mean per-CA pLDDT (B-factor column), rescaled to 0-100.

    esmatlas stores pLDDT as a 0-1 fraction, NVIDIA as 0-100; if max value
    <= 1 we scale by 100 to return the conventional 0-100 scale.
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
    ap.add_argument("--cif", action="store_true", help="Request mmCIF (esmatlas backend only)")
    ap.add_argument("--backend", choices=["auto", "esmatlas", "nvidia"], default="auto",
                    help="auto (default): esmatlas ≤400 aa, NVIDIA NIM for longer or on failure")
    args = ap.parse_args()

    name, seq = read_sequence(args.sequence)
    fmt = "cif" if args.cif else "pdb"
    out = Path(args.out) if args.out else Path(f"{name}.{fmt}")

    print(f"folding {name} ({len(seq)} aa) → {out}", file=sys.stderr)
    text, used = fold(seq, fmt=fmt, backend=args.backend)
    # NVIDIA only emits PDB; if --cif was requested but we fell back, fix suffix.
    if args.cif and used == "nvidia" and out.suffix == ".cif":
        out = out.with_suffix(".pdb")
    out.write_text(text)

    if out.suffix == ".pdb":
        plddt = mean_plddt_from_pdb(text)
        if plddt is not None:
            scale = "0-1 fraction" if used == "esmatlas" else "0-100"
            print(f"backend: {used} | B-factor scale: {scale} | mean pLDDT (0-100): {plddt:.1f}",
                  file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
