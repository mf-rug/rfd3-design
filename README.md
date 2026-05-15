# rfd3-design

Run [RosettaCommons Foundry](https://github.com/RosettaCommons/foundry)
(RFdiffusion3, RoseTTAFold3, ProteinMPNN/LigandMPNN) on **macOS Apple
Silicon**, packaged as a Claude Code skill.

This repo is also useful as a standalone install guide / wrapper script:
the contents of `INSTALL.md` document a working install procedure with
all the workarounds discovered through hands-on debugging that are **not**
covered by the upstream README.

## What's in here

| File | Purpose |
|------|---------|
| `INSTALL.md` | Step-by-step install on macOS, including all the workarounds the upstream README doesn't mention (broken MPS, broken Hydra CPU config, required `is_legacy_weights`, RF3 silent early-stopping at pLDDT=0.5, …) |
| `SKILL.md` | Usage guide for de novo backbone design with RFD3, sequence design with MPNN, and ESMFold self-consistency. Doubles as the Claude Code skill body (YAML frontmatter at the top). |
| `run_cpu.sh` | The mandatory CPU dispatch wrapper. macOS MPS crashes every model in the suite with Metal kernel asserts; this script monkey-patches `torch.backends.mps.is_available()` and dispatches `rfd3` / `rf3` / `mpnn` correctly. |
| `scripts/esmfold.py` | One-shot ESMFold structure prediction via the public Meta API. Used for MPNN self-consistency checks (RFD3 backbone → MPNN sequence → ESMFold prediction → compare). |

## Quick start

1. Read `INSTALL.md` and follow the install procedure. Result: a Python
   venv at `$RFD3_DIR/.venv/` and ~5.5 GB of model weights at
   `~/.foundry/checkpoints/`.
2. Clone this repo into `~/.claude/skills/rfd3-design/` (or symlink it
   from your Dropbox-synced Claude config).
3. `chmod +x run_cpu.sh`.
4. Read `SKILL.md` for usage examples.

## Why does this exist?

The upstream Foundry README claims macOS MPS support via the
[fnachon community fork](https://github.com/fnachon/foundry). In practice
every model in the suite crashes on MPS with a Metal kernel assert
(`MPSNDArrayGatherND`, `mps.reshape`), and the shipped `trainer=cpu`
Hydra config that would force CPU is broken (references a missing `af3`
parent). The wrapper script here is the minimum-viable workaround.

## Status

- ✅ RFD3 de novo backbone design — works on CPU.
- ✅ RF3 fold — works on CPU. Default `early_stopping_plddt_threshold=0.5`
  must be set to `0` or runs silently produce no output.
- ✅ ProteinMPNN / LigandMPNN sequence design — works on CPU.
- ✅ ESMFold via API for self-consistency checks.
- ❌ MPS / GPU acceleration — broken upstream.
- ❌ Multi-GPU training — out of scope; inference only.

## License

See upstream Foundry for model weights and code licensing.
This wrapper + install documentation: MIT.
