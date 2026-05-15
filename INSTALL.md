# Foundry / RFD3 / RF3 / MPNN — macOS Apple Silicon Install

This documents the working install procedure for the RosettaCommons Foundry
suite (RFdiffusion3, RoseTTAFold3, ProteinMPNN/LigandMPNN) on Apple Silicon,
including the workarounds discovered through hands-on debugging that are
**not** covered by the upstream README.

## When to read this

Read this in full **before** the first install on a new machine, or when
debugging an inference failure. The companion `SKILL.md` (run RFD3 backbone
design) assumes the install described here is already present and accessible
via the `$RFD3_DIR` environment variable.

---

## What the upstream README gets right

The macOS install recipe published in the [Foundry README](https://github.com/RosettaCommons/foundry)
(via the [fnachon community fork](https://github.com/fnachon/foundry)) cleanly
installs the Python packages and downloads model weights. Approximate time: 10
minutes plus weight download.

## What the upstream README does **not** cover

The published instructions stop at "weights are downloaded". Actually running
inference on Apple Silicon requires several non-obvious workarounds:

1. **MPS doesn't work** for any model in the suite, despite the fork's claim.
   Both RF3 and MPNN crash with Metal kernel asserts (`MPSNDArrayGatherND`,
   `mps.reshape`). `PYTORCH_ENABLE_MPS_FALLBACK=1` does not help — these are
   kernel asserts, not unsupported-op fallbacks. **All inference must run on
   CPU.**
2. **The shipped `trainer=cpu` Hydra config is broken** — it references a
   missing `af3` parent. The Hydra-level CPU override does not work; we have
   to monkey-patch `torch.backends.mps.is_available()` from a Python wrapper.
3. **There is no public Python API**. Imports like `from rfd3 import RFDiffusion3`
   (which one might guess from the package name) do not exist. Real entry
   points are `RFD3InferenceEngine`, `RF3InferenceEngine`,
   `MPNNInferenceEngine` — and the MPNN engine validates input on
   instantiation, so default `None` values for `is_legacy_weights` and
   `out_directory` raise immediately.
4. **`is_legacy_weights=True` is required** for every checkpoint in the
   default registry (`base-models`). Buried in a collapsed `<details>` block
   in the upstream README.
5. **RF3's default `early_stopping_plddt_threshold=0.5`** kills output
   silently when the run lands at exactly 0.50 (a float32 rounding artefact
   that's likely on macOS). Disable by setting it to 0.
6. **Hydra config-key paths matter.** The RF3 inference yaml uses
   `@package _global_`, so its keys (e.g. `early_stopping_plddt_threshold`)
   are at top level — not under `inference_engine.*` as one might expect from
   the file path.

---

## Prerequisites

- **Hardware**: Apple Silicon (arm64). Any M-series chip.
- **RAM**: 16 GB minimum. 5–6 GB is consumed by RF3 model weights + activations.
- **Disk**: ≥ 10 GB free for weights + venv.
- **Python**: 3.12.x (Homebrew or pyenv).
- **Git** and **curl** in `PATH`.
- **No conda required.**

---

## Install procedure

Pick an install root and export it. Everything else flows from this.

```bash
export RFD3_DIR="$HOME/Documents/Code/rfd3_mac"   # adjust to taste
mkdir -p "$RFD3_DIR" && cd "$RFD3_DIR"
```

> Persist `$RFD3_DIR` in `~/.bashrc` (or `~/.zshrc`), or set it in
> `~/.claude/settings.json` under `env`, so future shells / Claude Code
> invocations find the install.

### 1. Create venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch

```bash
pip install torch
```

PyPI `torch` ships with MPS support on Apple Silicon — no special index URL.
At time of writing this resolves to `torch 2.7.1`.

### 3. Install Foundry from the macOS community fork

```bash
pip install "rc-foundry[all] @ git+https://github.com/fnachon/foundry.git"
```

Pulls RFD3, RF3, ProteinMPNN/LigandMPNN, and AtomWorks. The `rf3`
cuEquivariance extra is automatically skipped on macOS. **Do not** use the
upstream `RosettaCommons/foundry` package — it lacks the macOS patches.

### 4. Download model weights

```bash
foundry install base-models
```

Downloads ~5.5 GB to `~/.foundry/checkpoints/`:
- `rfd3_latest.ckpt` (2.5 GB)
- `rf3_foundry_01_24_latest_remapped.ckpt` (2.8 GB)
- `proteinmpnn_v_48_020.pt` (6 MB)
- `ligandmpnn_v_32_010_25.pt` (10 MB)

Verify with `foundry list-installed`.

### 5. Clone the repo (for example inputs)

The published example JSONs reference test PDBs and MSAs that ship with the
repo, not with the pip package. Required if you want to run the upstream
quick-test commands.

```bash
cd "$RFD3_DIR"
git clone --depth=1 https://github.com/RosettaCommons/foundry.git foundry
```

### 6. The CPU-forcing wrapper

The mandatory workaround for the broken MPS path ships with this skill at
`~/.claude/skills/rfd3-design/run_cpu.sh`. If you cloned this repo into
`~/.claude/skills/rfd3-design/` (or have it Dropbox-synced into that
location), just make sure it's executable:

```bash
chmod +x ~/.claude/skills/rfd3-design/run_cpu.sh
```

Invoke it as:

```bash
~/.claude/skills/rfd3-design/run_cpu.sh rfd3 design ...
~/.claude/skills/rfd3-design/run_cpu.sh rf3  fold   ...
~/.claude/skills/rfd3-design/run_cpu.sh mpnn        ...
```

The wrapper sources `$RFD3_DIR/.venv/bin/activate`, monkey-patches
`torch.backends.mps.is_available()` to return `False`, then dispatches to
the right entrypoint: `rfd3`/`rf3` use Typer (`app()`), `mpnn` uses
argparse (`main()`).

---

## Verification

### Smoke test: imports + MPS detection

```bash
source "$RFD3_DIR/.venv/bin/activate"
python -c "import torch; import rfd3, rf3, mpnn; print('OK', torch.__version__)"
```

Should print `OK 2.7.1` (or similar).

### Smoke test: ProteinMPNN inference on 1L2Y

```bash
curl -sL https://files.rcsb.org/download/1L2Y.pdb -o /tmp/1L2Y.pdb
mkdir -p /tmp/mpnn_smoke
mpnn \
    --model_type protein_mpnn \
    --checkpoint_path "$HOME/.foundry/checkpoints/proteinmpnn_v_48_020.pt" \
    --is_legacy_weights True \
    --structure_path /tmp/1L2Y.pdb \
    --out_directory /tmp/mpnn_smoke \
    --number_of_batches 1 --batch_size 1
```

Expected: ~2 s on CPU, writes a designed sequence FASTA. **Note:** the bare
`mpnn` shell command auto-selects MPS and crashes — always go via
`~/.claude/skills/rfd3-design/run_cpu.sh mpnn …` (or use the Python API with `device="cpu"`).

### Smoke test: RF3 fold (~22 min on 16 GB Mac)

Confirms the full pipeline including checkpoint, transforms, and CIF output.

```bash
cd "$RFD3_DIR/foundry"   # required: JSON has a relative MSA path
~/.claude/skills/rfd3-design/run_cpu.sh rf3 fold \
    inputs="$(pwd)/models/rf3/tests/data/5vht_from_json.json" \
    out_dir="$RFD3_DIR/rf3_test_out" \
    early_stopping_plddt_threshold=0 \
    n_recycles=3 \
    diffusion_batch_size=1
```

Expected output in `$RFD3_DIR/rf3_test_out/5vht_from_json/`:
- `5vht_from_json_model.cif`
- `5vht_from_json_summary_confidences.json` — pTM should be `>0.8`
- `5vht_from_json_ranking_scores.csv`

A pTM of ~0.91 on this test confirms the install is working correctly.

---

## Performance expectations (M-series, 16 GB)

| Model | Task | Settings | Wall time |
|-------|------|----------|-----------|
| RF3 | fold ~200-residue dimer | `n_recycles=3`, `diffusion_batch_size=1` | ~22 min |
| ProteinMPNN | design 20-residue monomer | 1 batch, 1 sequence | ~2 s |
| RFD3 | de novo design | depends on `num_timesteps` | not yet benchmarked here |

PyTorch CPU backend uses BLAS/OpenMP and parallelizes across cores
(observed ~3 cores at ~280 % CPU during RF3 forward). Performance is
**memory-bound**, not flops-bound — closing other apps before a run helps.

---

## Known limitations

1. **No MPS / no GPU acceleration.** All inference is CPU-only. Plan accordingly.
2. **No `bfloat16`.** RF3 uses bfloat16 AMP automatically when running on CPU.
3. **No multi-GPU training.** Forks/PRs targeting macOS are inference-only.
4. **`rf3 fold` requires absolute paths** for input CIF files (per upstream README).
5. **Example JSONs use repo-relative paths** (e.g. for MSAs) — `cd` into
   `$RFD3_DIR/foundry` before running them, or rewrite the paths to absolute.
6. **`CCD_MIRROR_PATH` / `PDB_MIRROR_PATH` / `LOCAL_MSA_DIRS` warnings** at
   startup are harmless unless you have local PDB/CCD/MSA mirrors.

---

## Upstream references

- Foundry repo: https://github.com/RosettaCommons/foundry
- macOS fork: https://github.com/fnachon/foundry
- RFD3 input spec: https://rosettacommons.github.io/foundry/models/rfd3/input.html
- RF3 docs: https://rosettacommons.github.io/foundry/models/rf3/
