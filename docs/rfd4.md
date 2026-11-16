# RFD4-Proteina — install and first run (cluster + Mac)

**Status:** smoke-tested on both sides 2026-11-13/15. Not yet wired into any
generator tooling. This doc records what was measured, not a workflow.

RFD4-Proteina is a *separate model and separate repo* from the Foundry suite
that the rest of this skill covers — different inputs (YAML, not RFD3 JSON),
different CLI, different output convention. Do not assume anything from
`SKILL.md` carries over.

The parent skill's frontmatter names RFD4 as explicitly **not** covered, and
points here. That is deliberate: "RFD4" and "rfdiffusion 4" are lexically
adjacent to the RFD3 triggers, so without the clause the skill either misses
entirely (RFD4 handled from general knowledge) or — worse — matches on
similarity and applies RFD3 rules by analogy. The `.cif.gz` glob is the
obvious casualty, since that lesson is stated emphatically in `SKILL.md` and
is exactly wrong here. Loading a skill to deliver a redirect costs a few
hundred tokens; the alternative failure is silent and wrong.

- Repo: https://github.com/RosettaCommons/RFD4-Proteina
- Weights: `nvidia/RFD4-Proteina` on Hugging Face (~1.7 GB, 2 files)
- License: BSD 3-Clause

---

## The three things most likely to catch you out

1. **Outputs are plain, uncompressed `.cif`** — the exact opposite of RFD3's
   `.cif.gz`. Anyone who internalised "RFD3 writes `.cif.gz`, never glob for
   `.cif`" will silently miss every RFD4 result by applying that lesson.
   Path: `<out_dir>/inference_samples/<name>_<i>/<name>_<i>_generated.cif`.
2. **`hardware.accelerator` is ignored at inference.** `generate.py` hardcodes
   `accelerator="gpu"` in its `L.Trainer(...)` call. The config key exists and
   does nothing. There is therefore **no CPU fallback** — on a GPU-less Linux
   box inference fails rather than degrading.
3. **`DATA_PATH` must be exported even when it is unused.** `configs/inference.yaml`
   interpolates `${oc.env:DATA_PATH}` for `ckpt_path` and `autoencoder_ckpt_path`.
   Those lose to `hf_ckpt_path` at load time, but Hydra still has to *resolve*
   them, so an unset `DATA_PATH` fails the run before the model is reached.

---

## Part 1 — HPC cluster (Snellius)

Measured on Snellius, 2026-11-15.

| Step | Partition | Wall | Cost |
|---|---|---|---|
| install | `rome` (CPU) | 16:49 | ~4.5 SBU |
| smoke (quickstart binder) | `gpu_a100` | 2:42 (150 s generation) | ~5.8 SBU |

Peak VRAM **2,657 MiB of 40 GB**. The quickstart preset is nowhere near a full
A100 — if RFD4 ever gets a GPU-tier table it starts far lower than RFD3's, and
a MIG slice would likely serve for jobs this size.

### Install notes specific to the cluster

- **Use the `all-cuda13` extra.** The extra tracks the *driver*, not the card:
  Snellius reports driver 595.71.05 / CUDA 13.2 on both `gpu_a100` and
  `gpu_h100`. `all-cuda12` would be the degraded choice, not the safe one —
  it drops the BioIR runtime (`inference_backend=bioir`).
- **The pip bootstrap is required here, not optional.** The module ships pip
  24.0, inside the range upstream flags for the compromised eMudhra/emSign CA
  (October 2026). Upstream's `docs/installation.rst` gives a hash-pinned pip
  26.2.2 URL to install *first*, before anything else is downloaded. Follow it.
- **`uv` is not available and has no module.** Plain pip works (16:49 against
  upstream's "~10 min" claim), which avoids putting another tool onto shared
  storage. Do not `curl | sh` an installer on a shared system.
- **Set `HF_HOME` into project space.** Left at its default the 1.7 GB cache
  lands in the backed-up 200 GiB home and is re-fetched by every new user.
- **Compute nodes reach Hugging Face.** No pre-fetch needed: 2 files in ~3 s,
  unauthenticated, at the pinned revision. The "set `HF_TOKEN` for higher rate
  limits" line is advisory only.

### Reproduce

```bash
# venv at /projects/<proj>/rfd4_venv
export HF_HOME=/projects/<proj>/hf_cache
export DATA_PATH=/projects/<proj>/rfd4_data
rfd4-generate inference_experiment=quickstart inputs=binder.yaml out_dir=./runs/first
```

Job scripts live in `/projects/<proj>/rfd4_jobs/{install,smoke}/` on Snellius.

### Habrók: excluded

The RFD4 venv is **82,862 files — 16 % of the 512k soft inode cap** before
weights or outputs. That is the number to lead with if anyone asks. Same
reasoning as the Foundry venv (54,440 files); RFD4 is worse.

### Tooling

RFD4 needs **its own generator, in a sibling repo** — not `rfd3-setup-job`.
Its YAML-through-Hydra input model shares nothing with the RFD3 JSON builder,
its `extra="forbid"` field mirror, or the `+specification.*` prefix rule;
those would have to be rewritten, not parameterised. `hpcjob` needs **nothing
new** — it is model-agnostic, so submit/pull/preflight/doctor already work;
a `job.sh` with a `#SBATCH --job-name` is the whole requirement.

**Built, 2026-11-16: [mf-rug/rfd4-cluster](https://github.com/mf-rug/rfd4-cluster)**
(`rfd4-setup-job`). It authors the condition spec, resolves it into an
annotated CIF *locally* through RFD4's own validation pipeline before
submitting, sizes the job by worst-case token count, and emits a job.sh for
`hpcjob submit`. The §2 decision test gave two of four conditions, which is
the threshold rather than a margin -- the repo's README says so, and says when
to collapse it back into a hand-written job.sh.

Note it deliberately does **not** wrap upstream's `scripts/launch_generate.sh`.
That launcher is the better tool for job arrays and dependent metric
aggregation, but it submits from inside the cluster; this toolbox authors
locally and rsyncs. Use upstream's directly for batch fan-out.

---

## Part 2 — Local Mac (Apple Silicon)

**Upstream says macOS is unsupported. It ran anyway, on MPS.**

This is the opposite of the RFD3/RF3/MPNN situation documented in
`INSTALL.md`, where MPS crashes with Metal kernel asserts and everything must
be forced to CPU via `run_cpu.sh`. Do **not** reach for `run_cpu.sh` here.

Measured on an M-series iMac, 16 GB, 2026-11-13:

| | |
|---|---|
| Design | README binder example, 86-res binder vs 115-res target, 60 steps |
| Generation | 7 min 26 s |
| Wall (incl. weight download) | 8 min 40 s |
| Peak memory | 5.9 GB |
| CPU time | 31 s — i.e. the work ran on the GPU, not the CPU |

Evidence it used MPS: 31 s CPU over a 7.5 min run, plus PyTorch's own
`'pin_memory' ... not supported on MPS now` warning in the log. Combined with
the hardcoded `accelerator="gpu"` (point 2 above), Lightning resolved "gpu" to
MPS on this machine.

### Install

```bash
# Python must be >=3.12,<3.15. A newer system python3 will be rejected
# with "requires a different Python".
python3.12 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install ./repo          # base package only, no extras
```

The CUDA extras are all gated `sys_platform == 'linux'` in `pyproject.toml`,
so the base package resolves cleanly on arm64 (torch 2.15). At runtime it logs
`cuEquivariance is unavailable; using torch` and proceeds.

### Run

```bash
export DATA_PATH="$PWD/../data"     # required even though unused — see point 3
rfd4-generate inputs=../binder.yaml out_dir=../runs/first \
    num_replicates=1 '~metrics' compile_lengths=null \
    nn.attention_type=torch num_workers=0
```

`compile_lengths=null` skips the `torch.compile` warm-up. `'~metrics'` drops
the scoring stack (the AF2 / OpenFold3 refolding environments are **not** set
up on the Mac; `rfd4-evaluate` is untested there).

### What was verified

Geometry only, not design quality: 2 chains, no NaNs, CA–CA 3.81 Å mean /
3.97 Å max, 26 inter-chain CA pairs under 8 Å. No refolding, no scoring. This
shows the pipeline runs end to end — nothing about whether the binder is good.

---

## Benchmarks (2026-11-16)

Measured by sweeping binder length against the same 115-residue target, at
`nsteps=20` (peak memory is independent of step count, so a cheap 20-step
probe sizes a 60-step run correctly; `nsteps` has a floor of 20).

**Memory** -- clean quadratics on all three machines:

| Machine | fit (MiB) | dev | 85%-of-RAM ceiling |
|---|---|---|---|
| A100-40GB | `2077 + 0.01039 L^2` | 2.9% | ~1780 res |
| MacBook, 64 GB | `5257 + 0.01802 L^2` | 3.5% | ~1670 res |
| iMac M1, 16 GB | `4551 + 0.01989 L^2` | 2.8% | ~690 res |

**No OOM cliff was reached anywhere** -- every sweep ran out of range, not
memory. Treat the ceilings as fits extrapolated to the hardware.

**Speed** -- generation only, not wall time:

| Residues | A100 | MacBook 64 GB | iMac M1 16 GB |
|---|---|---|---|
| 300 | 23 s | 29 s | 315 s |
| 500 | 21 s | 71 s | 1322 s |
| 900 | 61 s | 247 s | -- |
| 1100 | ~85 s | 361 s | -- |

### What this means in practice

- **A MacBook is a viable RFD4 machine.** About 4x slower than an A100 at
  size, and within 30% of one at 300 residues. That is nothing like the RFD3
  situation, where the Mac path exists only for smoke tests.
- **The M1 iMac is not**, and the reason is mostly not the chip: it ran 11-19x
  slower than the other Mac while producing *the same* memory curve, with the
  gap widening with size (12.7x at 200 residues, 18.6x at 500). That is the
  signature of memory pressure -- it was already 9 GB into swap before the
  sweep began. Usable up to ~300 residues; painful beyond.
- **Generation time scales as L^1.73** over 700-1200 residues on the A100,
  not the L^2 RFD3 behaviour suggested.
- **`maximum resident set size` is useless on Apple silicon** -- it sat at
  2.3-2.6 GB while the real footprint reached 26 GB, a 10x understatement.
  Use "peak memory footprint" from `/usr/bin/time -l`.

The GPU tier table in [mf-rug/rfd4-cluster](https://github.com/mf-rug/rfd4-cluster)
is built on the A100 fit above.

## Provenance: RFD4 needs no manifest

`docs/checkpoint-manifest-schema.md` describes a checkpoint-provenance scheme
agreed for RFD3. **RFD4 does not need it.** It already records
`ckpt_path`, `ckpt_name` and `seed` per design in
`<out_dir>/inference_metrics/per_example_metrics.csv`, and
`configs/inference.yaml` pins `hf_ckpt_revision` to an exact Hub commit, so a
design is attributable without any sidecar of ours. That schema stays an
RFD3-only patch.
