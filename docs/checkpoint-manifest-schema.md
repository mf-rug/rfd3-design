# Checkpoint provenance — `checkpoint_manifest.json` (v1)

**Status:** agreed, **not implemented** (2026-11-13). Neither the Mac nor
cluster side writes this file yet. This doc exists so the agreement
survives one of us re-opening the question in six months.

## Motivation

RFD3 writes a per-design sidecar (top-level keys: `ckpt_path`,
`diffused_index_map`, `metrics`, `seed`, `specification`) that records
the **resolved absolute path** of the checkpoint used. That's good
for debugging on one machine, but it does not answer: *"were these two
batches, possibly on different machines, run under the same model?"*

Pinning by registry name (`rfd3`) is brittle — the registry entry resolves
to whatever `rfd3_latest.ckpt` happens to be on disk, which drifts whenever
anyone re-runs `foundry install base-models`. Recording the checkpoint's
SHA-256 in a sibling file gives us provenance without constraining anyone's
install.

The sidecar belongs to rc-foundry (upstream). We can't extend it without
patching upstream, so the manifest is a separate file our wrappers write.

## Schema

One `checkpoint_manifest.json` per job/output directory:

```json
{
  "schema": "foundry-checkpoint-manifest/1",
  "generated_utc": "2026-11-13T09:41:02Z",
  "checkpoints": {
    "rfd3_latest.ckpt": {
      "path": "/projects/<proj>/foundry/checkpoints/rfd3_latest.ckpt",
      "realpath": "/gpfs/work4/0/<proj>/foundry/checkpoints/rfd3_latest.ckpt",
      "sha256": "<64 hex chars>",
      "size": 2690316669
    },
    "ligandmpnn_v_32_010_25.pt": { "...": "..." }
  }
}
```

Field semantics:

- **Key** — checkpoint basename (e.g. `rfd3_latest.ckpt`). Basenames are
  the only identifier stable across machines; both the cluster install
  and the Mac install resolve `foundry install base-models` to files
  with the same basenames.
- **`path`** — what the config / CLI pointed at (may be a symlink).
- **`realpath`** — `os.path.realpath(path)`. This is what the RFD3
  sidecar's `ckpt_path` contains (verified on Snellius 2026-11-13: config
  `/projects/<proj>/…` resolved to `/gpfs/work4/0/<proj>/…`), so a reader
  joining sidecar→manifest keys off `realpath` with basename as fallback.
- **`sha256`** — SHA-256 of the file.
- **`size`** — bytes; catches a truncated download before anyone bothers
  comparing hashes.

## Rules that aren't obvious

1. **Hash only the checkpoints the job actually touched.** 7.9 GB of
   base-models hashes in ~12 s on Lustre, but most jobs touch only
   RFD3 + one MPNN variant (~4 s). Hashing all five every job is waste
   and produces misleading provenance (the job did not depend on the
   un-touched weights).
2. **Do not cache.** The measured wall time (3.84 s for `rfd3_latest.ckpt`,
   0.07 s for `ligandmpnn_v_32_010_25.pt` on Lustre; typically faster on
   local SSD) is cheap enough that an mtime-keyed cache adds bugs for no
   win.
3. **Do not join on `path`.** Our sidecars record `realpath`; a reader
   joining on the configured path will silently fail whenever a symlink
   sits between them. Join on `realpath`, fall back to basename.
4. **Do not patch the rc-foundry sidecar.** Keep the manifest separate.

## Fields intentionally out of v1

- **`foundry_version`** — each side sources it differently; add when a
  reproducibility question demands it.
- **host / username** — environment-specific, and the hash already answers
  the question that matters.

## Reproducibility question answered

Together with the rc-foundry sidecar, a design is attributable by:

```
sidecar.seed  +  manifest.checkpoints[<basename>].sha256  +  sidecar.specification
```

## When to implement

Neither side is blocked. Either side may implement first; the other will
match this schema byte-for-byte when a real question comes up. Changes to
the schema bump `"schema": "foundry-checkpoint-manifest/N"`.
