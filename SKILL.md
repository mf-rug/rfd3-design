---
name: rfd3-design
description: Generate de novo protein backbones with RFdiffusion3 (RFD3) on the local Mac. Use when the user wants to design a protein backbone, scaffold a motif, or run RFD3 / RFdiffusion3 / "rfdiffusion 3" inference. Runs CPU-only; expect minutes-to-hours per design depending on length and timesteps.
---

# rfd3-design — run RFdiffusion3 on this Mac

This skill runs RFdiffusion3 (RFD3) for de novo backbone generation against
the locally-installed Foundry suite. It assumes the install procedure in the
companion `INSTALL.md` has already been completed.

> **Read `INSTALL.md` on demand** if: the venv is missing, weights are not
> installed, the `run_cpu.sh` wrapper is absent, or any verification step
> fails. The install doc captures gotchas (broken MPS path, broken Hydra CPU
> config, required `is_legacy_weights`, etc.) that are not in the upstream
> README.

## Hard requirements

- `$RFD3_DIR` — install root. Default `$HOME/Documents/Code/rfd3_mac`;
  if unset, fall back to that path (`: "${RFD3_DIR:=$HOME/Documents/Code/rfd3_mac}"`).
- `$RFD3_DIR/.venv/` with `rfd3` importable.
- `~/.claude/skills/rfd3-design/run_cpu.sh` executable (ships with this skill).
- `~/.foundry/checkpoints/rfd3_latest.ckpt` (skip if MPNN/ESMFold only).

If missing in both env and default, read `INSTALL.md`.

## Performance reality on this hardware

- **CPU-only** (MPS is broken on macOS for this model — see `INSTALL.md`).
- Wall time scales roughly with `length × num_timesteps × diffusion_batch_size`.
  As a rough anchor: a single-batch RF3 fold of ~200 residues with 3 recycles
  takes ~22 min on a 16 GB M-series Mac. RFD3 timing depends on
  `num_timesteps` (default 200 — much longer than RF3's recycles).
- Always **start with a short test run** (small `length`, small
  `num_timesteps`, `diffusion_batch_size=1`) to confirm the design compiles
  before queuing anything substantial.
- Before kicking off a long run, **always confirm the parameters with the
  user** — the user will lose the machine for the duration.

## Authoring the input JSON

RFD3 takes a JSON (or YAML) where each top-level key is one independent
design configuration. The simplest backbone-only request:

```json
{
    "monomer_100": {
        "length": 100
    }
}
```

For a length range (uniform random per design):

```json
{
    "monomer_80_120": {
        "length": "80-120"
    }
}
```

Multiple specs in one file, each independent:

```json
{
    "small": { "length": 60 },
    "medium": { "length": 120 }
}
```

For motif scaffolding, binders, partial diffusion, symmetry, ligand
conditioning, hydrogen-bond conditioning, hotspots, etc., consult the
[full input specification](https://rosettacommons.github.io/foundry/models/rfd3/input.html)
or the JSON examples shipped at `$RFD3_DIR/foundry/models/rfd3/docs/examples/`.

## Running a design

Save the JSON somewhere convenient (e.g. `$RFD3_DIR/jobs/<name>.json`) and run:

```bash
~/.claude/skills/rfd3-design/run_cpu.sh rfd3 design \
    inputs="$RFD3_DIR/jobs/<name>.json" \
    out_dir="$RFD3_DIR/jobs/<name>_out" \
    diffusion_batch_size=1 \
    inference_sampler.num_timesteps=50 \
    prevalidate_inputs=True \
    skip_existing=False
```

**Why these arguments:**

| Argument | Why |
|----------|-----|
| `run_cpu.sh rfd3 design …` | Mandatory CPU wrapper — bypasses broken MPS path |
| `inputs=…` | Path to the JSON (absolute is safest) |
| `out_dir=…` | Output directory (auto-created) |
| `diffusion_batch_size=1` | One design per batch — the floor for CPU runs |
| `inference_sampler.num_timesteps=50` | Reduced from default 200 — faster smoke test; **raise to 200 for real designs** |
| `prevalidate_inputs=True` | Catches malformed JSON before loading the 2.5 GB checkpoint |
| `skip_existing=False` | Force re-run; default `True` skips files that exist already |

**Quick debug without a JSON file** (uses defaults for everything):

```bash
~/.claude/skills/rfd3-design/run_cpu.sh rfd3 design \
    inputs=null \
    out_dir="$RFD3_DIR/jobs/quick_debug" \
    specification.length=60 \
    diffusion_batch_size=1 \
    inference_sampler.num_timesteps=20 \
    skip_existing=False
```

This is the fastest possible end-to-end check that the model loads and runs.

## Outputs

Per design, RFD3 writes (under `out_dir`):

- `<spec_name>_<idx>.pdb` — designed structure
- `<spec_name>_<idx>.json` — the resolved input spec + metadata (sampled
  contig, sequence, atom/residue counts)
- `<spec_name>_<idx>_traj.pdb` — diffusion trajectory (only if
  `dump_trajectories=True`)

Inspect a design with `pymol` / `chimerax` / VS Code molstar.

## After a run

- Designed backbones are sequence-less. Run ProteinMPNN / LigandMPNN via the CPU wrapper:
  ```bash
  ~/.claude/skills/rfd3-design/run_cpu.sh mpnn \
      --model_type protein_mpnn \
      --checkpoint_path ~/.foundry/checkpoints/proteinmpnn_v_48_020.pt \
      --is_legacy_weights True \
      --structure_path /abs/path/to/design.pdb \
      --out_directory /abs/path/to/seq_out \
      --number_of_batches 1 --batch_size 1
  ```
  Ligand-aware: `--model_type ligand_mpnn --checkpoint_path ~/.foundry/checkpoints/ligandmpnn_v_32_010_25.pt`.

  **Gotchas** (rc-foundry CLI, NOT the dauparas/LigandMPNN repo):
  - Flags are `--structure_path` / `--out_directory` — **not** `--pdb_path` / `--out_folder`.
  - Two `CCD_MIRROR_PATH` / `PDB_MIRROR_PATH` "not set" warnings always print. Harmless for protein-only redesign; ignore.
  - Run is nearly silent on success — verify by `ls` on `--out_directory`.
  - Outputs: structures as **`<name>_b<batch>_d<design>.cif`** (CIF, not PDB); sequences pooled into one **`<name>.fa`** with `sequence_recovery=...` in each header. Use `--write_structures False` to skip CIFs.
  - Fixed positions / scoring: see [LigandMPNN docs](https://github.com/dauparas/LigandMPNN).
- For structure validation, fold the designed sequence and compare to
  the RFD3 backbone. Fastest path: the `scripts/esmfold.py` helper
  (see below). For longer sequences or complexes, use RF3 once those
  weights are installed.

## Quick structure prediction (ESMFold via API)

`scripts/esmfold.py` calls Meta's public ESMFold endpoint
(`api.esmatlas.com/foldSequence/v1/{pdb,cif}`) — POST raw sequence as
`text/plain`, no auth, no install, no GPU. Typical response: a few
seconds for sequences up to ~400 residues.

```bash
# Sequence on the command line
python ~/.claude/skills/rfd3-design/scripts/esmfold.py \
    MKWVTFISLLLLFSSAYS -o /tmp/test.pdb

# Or a FASTA file (uses the first record's header as the name)
python ~/.claude/skills/rfd3-design/scripts/esmfold.py /path/to/design.fa
```

The output PDB has per-residue **pLDDT in the B-factor column as a
0–1 fraction** (note: not the 0–100 scale that AlphaFold/RF3 PDBs use —
multiply by 100 if you want to compare directly, or pass the raw
column to `ColorBFactor*` with `first=0.5, last=0.9`). The script
prints **mean CA pLDDT rescaled to 0–100** to stderr (>90 very high,
70–90 confident, 50–70 low, <50 disordered).

**Primary use case in this pipeline — MPNN self-consistency check:**

1. Run RFD3 → backbone PDB.
2. Run ProteinMPNN/LigandMPNN on the backbone → designed FASTA.
3. Run `esmfold.py` on each FASTA record → predicted structure.
4. Compare predicted structure to the original backbone (e.g. CA-RMSD
   via `tmtools`, `pymol`, or YASARA `SupAtom`). High mean pLDDT and
   low RMSD ⇒ the designed sequence likely folds to the intended
   backbone.

**Limits:** public endpoint, occasionally rate-limited or down;
sequences much longer than ~400 aa often 400/timeout; monomer only
(no complexes, no MSA conditioning). For longer sequences or
complexes, fall back to RF3 once those weights are installed
(`foundry install rf3`).

The helper only needs `requests` (already in the rfd3 venv).

## When to stop and ask

- The user has not specified `length` or any constraints.
- The user wants `num_timesteps > 100` and `diffusion_batch_size > 1`
  on this 16 GB / CPU-only machine — confirm they are willing to wait.
- The output directory already contains designs with `skip_existing=True`
  set — clarify whether to overwrite or skip.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `MPSGraph` / `mps.reshape` / Metal kernel assert | Forgot the `run_cpu.sh` wrapper | Re-run via `~/.claude/skills/rfd3-design/run_cpu.sh rfd3 …` |
| `Could not override 'X'` from Hydra | Wrong key path (config is `@package _global_`) | Check the actual key in the relevant YAML under `$RFD3_DIR/.venv/lib/python3.12/site-packages/rfd3/configs/` |
| `MissingConfigException: trainer/af3` | Tried to use `+trainer=cpu` Hydra override | Don't. Use `run_cpu.sh` instead |
| Empty `out_dir` after a run that printed no error | `skip_existing=True` and the file already existed, or RF3-style early-stopping fired | Check the output log for "skipping" / "early stopping"; rerun with `skip_existing=False` |
| Run hangs at "Predicting structure 1/1" | Normal — single forward pass through 2.5 GB model on CPU is silent for many minutes | Wait, or kill if it exceeds your timeout |

## Future extensions

This skill currently covers only de novo backbone generation with RFD3.
The CPU wrapper supports `rfd3`, `rf3`, and `mpnn` — but this skill does
not yet codify recommended workflows for:

- An opinionated ProteinMPNN / LigandMPNN sequence-design pipeline on
  RFD3 outputs (batch sampling, fixed-position handling, scoring).
- RF3 self-consistency validation of designed sequences.
- Motif scaffolding workflows (active sites, binders) with worked examples.
- Symmetric oligomer design.
- Hydrogen-bond–conditioned design (requires HBPLUS install).
