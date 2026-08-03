---
name: rfd3-design
description: Design proteins with RFdiffusion3 (RFD3) and ProteinMPNN/LigandMPNN — de novo backbones, motif scaffolding, binders, symmetric oligomers, plus sequence design for them. Use when the user wants to design/scaffold/diffuse a protein backbone, run RFD3 / RFdiffusion3 / "rfdiffusion 3", or run MPNN inverse folding. Covers both the GPU HPC cluster (default for real work) and the local CPU-only Mac (smoke tests).
---

# rfd3-design — RFdiffusion3 and MPNN

Two places to run, with very different economics. **Pick deliberately.**

| | HPC cluster (GPU) | Local Mac (CPU) |
|---|---|---|
| Use for | anything real — production runs, batches, long designs | smoke tests, tiny one-offs, when the cluster is down |
| Speed | GPU minutes | **CPU-only**; a 200-residue job takes tens of minutes to hours |
| Tooling | `rfd3-setup-job` + `hpcjob` | `run_cpu.sh` wrapper |
| Section | Part 1 below | Part 2 below |

**Default to the cluster** unless the user asks for local, or the design is
tiny and they want an answer in this session.

Both paths share the same input-JSON format — see "Authoring the input JSON",
which applies to each.

---

# Part 1 — Running on the HPC cluster

Two local CLI tools, both **multi-cluster** (`--cluster NAME`; without it the
registry's `default_cluster` is used). This mirrors the `boltz-hpc` toolbox
exactly, and shares its transport tool.

| Step | Tool | What it does |
|------|------|--------------|
| 1. Generate | `rfd3-setup-job` | Writes the RFD3 design JSON + a **cluster-correct** `job.sh` locally |
| 2. Submit / fetch | `hpcjob` | `hpcjob submit` uploads + `sbatch`; `hpcjob pull` rsyncs results back |

Repo: `$HOME/Documents/code/rfd3_cluster` (see its `README.md` and `CLAUDE.md`).
Cluster settings live in `~/.config/rfd3-setup/config.yaml` — modules, venv,
checkpoint dir, jobs dir, GPU tiers, partitions, per cluster. Nothing
cluster-specific is hardcoded, so adding a cluster is a config edit.

> **Before submitting anything**, confirm with the user what to design and with
> which settings. Design jobs consume real GPU allocation and take minutes to
> hours.

## Generating a job

```bash
# 20 unconditional backbones, 100 residues
rfd3-setup-job --length 100 --num-designs 20 --name denovo --out-dir ./denovo/

# length screen — three independent specs in one job
rfd3-setup-job --length "80|120|160" --num-designs 8 --name lenscan --out-dir ./lenscan/

# binder design with hotspots, then sequences for every backbone
rfd3-setup-job --input target.pdb --contig "A1-100,50-150" \
    --hotspots "A45-50,A75" --num-designs 32 --mpnn \
    --name binder --out-dir ./binder/

# motif scaffolding
rfd3-setup-job --input site.pdb --unindex "A431,A572-573" --length "120-180" \
    --num-designs 16 --name motif --out-dir ./motif/

# partial diffusion around an existing structure
rfd3-setup-job --input start.cif --partial-t 10 --set 'unindex=A20-40' \
    --num-designs 8 --name partial --out-dir ./partial/

# MPNN alone, over backbones already on the cluster
rfd3-setup-job --mpnn-only \
    --structures remote:/scratch/$USER/rfd3_jobs/denovo/output/rfd3 \
    --mpnn-seqs 8 --name reseq --out-dir ./reseq/
```

Key flags: `--num-designs N` (split into `diffusion_batch_size × n_batches`),
`--timesteps` (default 200), `--seed`, `--mpnn` / `--mpnn-only`,
`--mpnn-model {protein_mpnn,ligand_mpnn}`, `--mpnn-seqs`, `--cluster`,
`--partition` / `--time` / `--gpu` / `--mem` to override the recommendation.

**Anything the flags don't cover** (symmetry, h-bond conditioning, RASA burial,
per-atom selections): hand-write the inputs JSON and pass `--spec-json FILE`,
or set one field inline with `--set key=value` (values parsed as JSON where
possible). `rfd3-setup-job --print-fields` lists every accepted field. Don't
invent flags — the escape hatch is the intended route.

## Submitting and fetching

```bash
hpcjob submit ./denovo/job.sh --cluster habrok --jobs-dir /scratch/$USER/rfd3_jobs
hpcjob status <job_id> --cluster habrok
hpcjob pull denovo --cluster habrok --yes
```

`rfd3-setup-job` prints the exact submit line (including `--jobs-dir`) when it
finishes — use that rather than composing one by hand. Keep the **same
`--cluster`** across generate → submit → pull: the job script bakes in that
cluster's module/venv/partition and will fail elsewhere.

Results land in `<out-dir>/output/rfd3/` (backbones + per-design JSON),
`<out-dir>/output/mpnn/` (`.fa` sequences + designed `.cif`), plus a
`<name>_COMPLETED_<id>.log` summary written by the job's exit trap.

## Cluster-side gotchas

- **RFD3 output is gzipped mmCIF (`*.cif.gz`), never PDB**, plus a sibling
  `.json`. Don't glob for `.cif`/`.pdb` and conclude a run failed. **Keep the
  JSON** — it carries free per-design QC (`n_chainbreaks`, interresidue clash
  counts, `helix_fraction`, `radius_of_gyration`, `sampled_contig`), which is
  the right pre-filter before spending MPNN or folding time on a backbone. The
  summary log reports a clean/flagged count.
- **Memory is set by design length (~L²), not batch size.** Measured at 200
  timesteps: L100 3.4 GB, L200 8.0 GB, L400 26 GB, L600 57 GB — while batch
  1 → 8 costs ~1% more VRAM for 8× the designs. **Prefer large
  `--num-designs` over more jobs**, and treat the tier's `max_length` as a hard
  cliff: RFD3 OOMs abruptly on one big allocation, so a slightly-too-long design
  dies in seconds rather than running slowly.
- **Hydra: `specification.*` overrides need a leading `+`, everything else must
  not have one.** `--rfd3-set` applies this automatically; only relevant if you
  hand-write an `rfd3 design` command line.
- **Weights are not in `$HOME`.** Foundry searches `~/.foundry/checkpoints`
  plus `$FOUNDRY_CHECKPOINT_DIRS`; the job scripts export the latter because
  the checkpoints live on scratch/project. There is **no fallback** — foundry
  writes nothing to `$HOME` — so a "checkpoint not found" failure is almost
  always a missing export, not a missing file.
- **MPNN runs as one process over all backbones**, via its `--config_json`
  batch interface (built on the cluster by `python -m rfd3_tools mpnn-config`,
  since the backbone filenames don't exist at generation time). Never loop one
  `mpnn` process per structure — model load dominates.
- **MPNN checkpoint filenames carry version numbers** (`proteinmpnn_v_48_020.pt`),
  so the job resolves them by glob. If a run fails to find weights, check the
  glob matched something in the configured checkpoint dir.
- **Which cluster is actually usable.** Snellius is installed and verified
  (A100 / H100 / cheap A100-MIG tier); **Hábrók has no Foundry install yet**,
  and its config block holds a deliberate `PLACEHOLDER-Python-3.12` module
  because `rc-foundry` requires Python `>=3.12,<3.13` and the 3.11.5 Boltz
  module cannot be reused. Generating a Hábrók job will produce a script that
  cannot run until that install happens.
- **Cores default to the partition's QOS per-GPU allowance.** GPU nodes are
  charged per GPU, so fewer cores saves nothing — and exceeding the ceiling
  makes Slurm reject the job outright (`QOSMaxCpuPerJobLimit`).
- **Tier/walltime numbers are measured on Snellius, extrapolated on Hábrók.**
  Recalibrate `gpu_tiers` / `timing` in `~/.config/rfd3-setup/config.yaml`
  rather than patching code.
- If the cluster install is missing or broken, `HANDOFF.md` in the repo is the
  install procedure, and its §1 lists the traps already hit once (Python 3.12
  requirement, the `+specification` syntax, `.cif.gz` output, 7.9 GB of weights).

---

# Part 2 — Running on the local Mac (CPU-only)

Runs RFD3 against the locally-installed Foundry suite. Assumes the install
procedure in the companion `INSTALL.md` has been completed.

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

**Applies to both paths.** On the cluster, `rfd3-setup-job` writes this file
for you from flags (or takes yours via `--spec-json`, validating it before
submission); locally you write it by hand.

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

The authoritative field list is the pydantic model
`rfd3.inference.input_parsing.DesignInputSpecification`, declared with
`extra="forbid"` — **an unknown key is a hard runtime error**, so a typo costs a
whole run. `rfd3-setup-job --print-fields` prints the accepted names, and
validates any spec (flag-built or `--spec-json`) before submission.

Required-field rules worth knowing, since they're the usual cause of a rejected
spec: you need at least one of `input` / `contig` / `length`; if `input` is set
it must be *consumed* by `contig`, `unindex`, `ligand` or `partial_t`; and
`unindex` needs `contig` or `length` alongside it unless doing partial
diffusion.

## Running a design locally

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

`scripts/esmfold.py` has **two backends and picks automatically** (`--backend
{auto,esmatlas,nvidia}` to force):

- **esmatlas** (default for ≤400 aa) — Meta's public endpoint
  (`api.esmatlas.com/foldSequence/v1/{pdb,cif}`), POST raw sequence as
  `text/plain`, no auth, no install, no GPU. A few seconds for short
  sequences, but **caps at ~400 residues** (longer → 413/400/timeout).
- **nvidia** (automatic fallback) — NVIDIA's hosted ESMFold NIM
  (`health.api.nvidia.com/v1/biology/meta/esmfold`, same model, GPU),
  accepts up to **1024 aa**. Used automatically when the sequence is
  **> 400 aa**, or when esmatlas errors out. Needs a free `nvapi-...` key
  (https://build.nvidia.com → any model → "Get API Key") in
  `NVIDIA_API_KEY` or `~/.nvidia/.env` (`NVIDIA_API_KEY=nvapi-...`).
  PDB only (no `--cif`). This is why the old esmatlas-only path 413'd on
  large enzymes — the NVIDIA fallback now handles them transparently.

```bash
# Sequence on the command line
python ~/.claude/skills/rfd3-design/scripts/esmfold.py \
    MKWVTFISLLLLFSSAYS -o /tmp/test.pdb

# A FASTA file (uses the first record's header as the name).
# >400 aa → routed to the NVIDIA NIM backend automatically.
python ~/.claude/skills/rfd3-design/scripts/esmfold.py /path/to/design.fa
```

The output PDB has per-residue **pLDDT in the B-factor column**, but the
**scale differs by backend**: esmatlas writes a **0–1 fraction** (not the
0–100 scale AlphaFold/RF3 PDBs use — multiply by 100 to compare, or pass
the raw column to `ColorBFactor*` with `first=0.5, last=0.9`), while the
NVIDIA backend writes the conventional **0–100**. The script's stderr line
reports which backend ran and the file's B-factor scale, and prints
**mean CA pLDDT rescaled to 0–100** (>90 very high, 70–90 confident,
50–70 low, <50 disordered).

**Primary use case in this pipeline — MPNN self-consistency check:**

1. Run RFD3 → backbone PDB.
2. Run ProteinMPNN/LigandMPNN on the backbone → designed FASTA.
3. Run `esmfold.py` on each FASTA record → predicted structure.
4. Compare predicted structure to the original backbone (e.g. CA-RMSD
   via `tmtools`, `pymol`, or YASARA `SupAtom`). High mean pLDDT and
   low RMSD ⇒ the designed sequence likely folds to the intended
   backbone.

**Limits:** esmatlas is occasionally rate-limited or down and caps at
~400 aa; the NVIDIA fallback caps at 1024 aa and needs the free key above.
Both backends are **monomer only** (no complexes, no MSA conditioning).
For sequences > 1024 aa or for complexes, fall back to RF3 once those
weights are installed (`foundry install rf3`).

The helper only needs `requests` (already in the rfd3 venv).

## When to stop and ask

- The user has not specified `length` or any constraints.
- **Before submitting any cluster job** — confirm the design spec and the
  number of designs. GPU allocation is real and shared.
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

Covered now: de novo backbones, motif scaffolding, binders with hotspots,
partial diffusion, and batched MPNN sequence design — on the cluster via
`rfd3-setup-job`, locally via `run_cpu.sh`. Not yet codified:

- **RF3 self-consistency validation as a cluster stage.** Locally, ESMFold via
  `scripts/esmfold.py` is the fast substitute (monomer-only, ≤1024 aa); the
  cluster venv has RF3, but no job template wires it in yet.
- Symmetric oligomer design (reachable today via `--spec-json` with a
  `symmetry` block; no worked example here).
- Hydrogen-bond–conditioned design (requires HBPLUS install).
- Fixed-position / bias handling for MPNN (`fixed_residues`, `bias`,
  `omit_per_residue` — all exist in MPNN's per-input config but aren't exposed
  as flags).
- Scoring and ranking designs (no opinionated filter chain yet).
