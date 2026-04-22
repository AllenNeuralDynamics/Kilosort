# Applying frozen templates to a long recording

`kilosort.apply_templates` runs only the basis-pursuit (matching-pursuit)
stage of Kilosort4 against a fixed template bank `Wall` loaded from a prior
`run_kilosort` run. It skips drift correction, universal-template detection,
clustering, and merging. Cluster IDs are identical to those produced during
training.

Use this when a full sort on the long recording is infeasible but a short
training window is representative.

## Requirements

The training run must have been performed with:

- `settings['nblocks'] = 0` — no drift correction (chronic probes only).
- `save_extra_vars=True` — so `Wall.npy` is written alongside `ops.npy`.

The target recording must match the training recording in:

- channel count (`n_chan_bin`)
- sampling rate (`fs`)
- dtype (`data_dtype`)
- probe geometry (chanMap, xc, yc)

`apply_templates` inherits all of the above from the training `ops.npy`.

## Two-pass workflow

### 1. Train on a 1–2 hr window

```python
import numpy as np
from kilosort import run_kilosort

settings = {
    'n_chan_bin': 385,
    'nblocks': 0,              # REQUIRED: disables drift
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

- `ValueError: apply_templates requires training with nblocks=0 ...` — the
  training run used drift correction. The saved `dshift` is indexed by the
  training run's batch count and cannot be applied to a recording with a
  different duration. Re-train with `nblocks=0`.
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
- **Memory.** `tF` is held in RAM for the full recording before `save_to_phy`
  is called. A 24-hr recording with ~200 units firing at ~10 Hz at default
  `nearest_chans=10` and `n_pcs=6` needs ~40 GB host memory. If this is a
  blocker, split the recording into chunks and run `apply_templates` on each
  with different `tmin`/`tmax`, then concatenate the phy outputs manually.

## Implementation pointers

- `kilosort/run_kilosort.py` — `apply_templates`
- `kilosort/template_matching.py:56` — `extract(ops, bfile, U, device)` is
  the matching-pursuit entry point. `U` is the frozen template bank; the
  loop at `template_matching.py:194` does not update it.
- `kilosort/io.py:968` — `BinaryFiltered` reads the raw binary, applies
  highpass and whitening. With `dshift=None` the drift branch at
  `io.py:1020` is skipped.
- `kilosort/io.py:248` — `save_to_phy` writes the phy output; agnostic to
  how `clu` was produced.
