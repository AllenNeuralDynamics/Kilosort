# Applying frozen templates to a long recording

`kilosort.apply_templates` runs only the basis-pursuit (matching-pursuit)
stage of Kilosort4 against a fixed template bank `Wall` loaded from a prior
`run_kilosort` run. It skips universal-template detection, clustering, and
merging. Cluster IDs are identical to those produced during training.

Use this when a full sort on the long recording is infeasible but a short
training window is representative.

## Requirements

The training run must have been performed with:

- `save_extra_vars=True` — so `Wall.npy` is written alongside `ops.npy`.

The target recording must match the training recording in:

- channel count (`n_chan_bin`)
- sampling rate (`fs`)
- dtype (`data_dtype`)
- probe geometry (chanMap, xc, yc)

`apply_templates` inherits all of the above from the training `ops.npy`.

### Drift

Drift handling is inherited from training via `ops['nblocks']`:

- **`nblocks = 0` (no drift)** — recommended for chronic probes with
  negligible drift. Neither training nor apply does drift correction.
- **`nblocks >= 1` (drift)** — training estimates drift on the 1–2 hr
  window; apply independently re-estimates drift on the target recording
  using its own average fingerprint as the reference. The training
  reference frame and the target reference frame are not identical, but
  for probes where drift within a session is much less than
  `binning_depth` (default 5 µm), templates still match. Apply writes
  `drift_amount.png` and `drift_scatter.png` to the results directory
  for diagnostics.

## Two-pass workflow

### 1. Train on a 1–2 hr window

```python
import numpy as np
from kilosort import run_kilosort

settings = {
    'n_chan_bin': 385,
    'nblocks': 0,              # 0 = no drift (chronic probe); >=1 enables drift
    'tmin': 0.0,
    'tmax': 7200.0,            # 2-hour training window
    # any other settings you normally use
}

run_kilosort(
    settings=settings,
    probe_name='neuropixPhase3B1_kilosortChanMap.mat',
    filename='/path/to/full_24hr.bin',
    results_dir='train_out',
    save_extra_vars=True,      # REQUIRED: writes Wall.npy
)
```

After this finishes, `train_out/` contains the usual phy output plus
`ops.npy`, `Wall.npy`, `tF.npy`, `full_st.npy`, `full_clu.npy`, `full_amp.npy`.
Only `ops.npy` and `Wall.npy` are needed for the apply pass.

### 2. Apply frozen templates to the full recording

```python
import numpy as np
from kilosort import apply_templates

apply_templates(
    training_dir='train_out',
    filename='/path/to/full_24hr.bin',
    results_dir='apply_out',
    tmin=0.0,
    tmax=np.inf,                # full recording
)
```

`apply_out/` contains phy-formatted spike times, cluster assignments, and
amplitudes for the 24-hr recording. Cluster IDs match `train_out/`.

### 2b. Chunked apply (bounded host RAM)

The single-call version holds `tF` for the whole recording in RAM
(~40 GB for 24 hrs with default settings). For large recordings, use
`apply_templates_chunked`, which runs `apply_templates` on time chunks,
writes each to a temporary phy dir, and merges them into one:

```python
from kilosort import apply_templates_chunked

apply_templates_chunked(
    training_dir='train_out',
    filename='/path/to/full_24hr.bin',
    chunk_hours=4.0,              # one chunk ≈ 1/6 of 24 hrs → ~7 GB peak
    results_dir='apply_out',
    tmin=0.0, tmax=np.inf,
)
```

Peak RAM is bounded by a single chunk. Merge is mmap-backed for
`pc_features.npy`. Merged output is one standard phy directory — the
`_chunk_*` temp dirs are deleted unless `keep_chunks=True`.

Multi-file recordings work transparently (`BinaryRWFile` concatenates a
list of `filename`s into one virtual recording). To align chunk
boundaries with physical file boundaries:

```python
# Compute per-file durations, build explicit ranges.
import kilosort.io as kio
fs = 30000
ops = np.load('train_out/ops.npy', allow_pickle=True).item()
durations = [kio.get_total_samples(f, ops['n_chan_bin'], ops['data_dtype']) / fs
             for f in file_list]
starts = np.concatenate([[0], np.cumsum(durations)[:-1]])
chunk_ranges = [(float(s), float(s + d)) for s, d in zip(starts, durations)]

apply_templates_chunked(
    training_dir='train_out', filename=file_list,
    chunk_ranges=chunk_ranges, results_dir='apply_out',
)
```

### 2c. Stitch training (opt-in)

By default, the chunked wrapper re-extracts spikes across the entire
requested range (including the training window) for uniform treatment.
If you specifically want to preserve training's saved spike train and
only extract the rest, use `stitch_training=True`:

```python
apply_templates_chunked(
    training_dir='train_out',
    filename='/path/to/full_24hr.bin',
    chunk_hours=4.0,
    stitch_training=True,         # skip the training window in chunk schedule
    results_dir='apply_out',
)
```

This carves `train_out/ops.npy`'s `tmin`/`tmax` out of the chunk
schedule and includes `train_out/` as one of the inputs to the merge.
Be aware that training spikes were extracted with first-pass templates
while apply chunks use the frozen final templates, so the merged output
will have a small methodological discontinuity at the train/apply
boundary. Prefer the default unless you've manually curated
`train_out/`.

## Sanity check: apply on the training window

Before trusting the apply pass on 24 hrs, run it on the training window and
compare against the training output:

```python
apply_templates(training_dir='train_out', filename='/path/to/full_24hr.bin',
                results_dir='apply_train_window', tmin=0.0, tmax=7200.0)
```

Expected: spike counts per cluster close to (but not identical to) the
training spike counts — a few percent difference is normal because the final
`Wall` has not actually been used for extraction in the standard KS4 pipeline.
Spike-time agreement within ±0.5 ms should exceed 90% for clusters labeled
`good` in `train_out/cluster_KSLabel.tsv`. Large (>20%) deviations indicate a
configuration mismatch (wrong `Wrot`, `wPCA`, `nt0min`, etc.) — inspect the
`ops.npy` comparison.

## Common errors

- `FileNotFoundError: Wall.npy not found in ...` — training was run without
  `save_extra_vars=True`. Re-train with the flag set.
- `FileExistsError: filename '...' does not exist` — the target binary path
  is wrong.

## Limitations

- **New units are invisible.** Template matching can only find spikes from
  units present in the training window. Any unit that is silent during
  training or first appears later will not be detected. If you suspect unit
  turnover, pick a training window from the middle of the recording or sort
  multiple shorter training windows separately.
- **Noise statistics drift.** The whitening matrix `Wrot` is computed from
  the training window. If ambient noise covariance drifts over 24 hrs, later
  batches are slightly miswhitened. Usually small; if needed, recompute
  `Wrot` on the full recording as a follow-up (not currently exposed).
- **Drift reference-frame mismatch.** When `nblocks >= 1`, apply-pass
  drift is estimated against the target recording's own average
  fingerprint, not the training window's. If the two references differ by
  more than a few µm (e.g., sessions days apart), templates will be
  spatially offset. For same-session chronic recordings this is
  negligible; for cross-session reuse, a different approach is needed.
- **Memory (single-call).** `apply_templates` holds `tF` in RAM for the
  full recording before `save_to_phy`. A 24-hr recording with ~200 units
  firing at ~10 Hz at default `nearest_chans=10` and `n_pcs=6` needs
  ~40 GB host memory. For large recordings, use `apply_templates_chunked`
  (Section 2b) — peak RAM is bounded by a single chunk.

## Implementation pointers

- `kilosort/run_kilosort.py` — `apply_templates`, `apply_templates_chunked`,
  `_merge_phy_chunks`.
- `kilosort/template_matching.py:56` — `extract(ops, bfile, U, device)` is
  the matching-pursuit entry point. `U` is the frozen template bank; the
  loop at `template_matching.py:194` does not update it.
- `kilosort/io.py:968` — `BinaryFiltered` reads the raw binary, applies
  highpass and whitening. With `dshift=None` the drift branch at
  `io.py:1020` is skipped.
- `kilosort/io.py:248` — `save_to_phy` writes the phy output; agnostic to
  how `clu` was produced.
- `kilosort/postprocessing.py:10` — `remove_duplicates(spike_times,
  spike_clusters, dt)` used for cross-chunk dedup in the merge.
