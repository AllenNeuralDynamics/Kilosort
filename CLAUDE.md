# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository context

This is the [MouseLand/Kilosort](https://github.com/MouseLand/Kilosort) upstream codebase, checked out inside the ccg-analysis-monorepo. Remotes:

- `origin` → AllenNeuralDynamics/Kilosort (our fork — push here)
- `upstream` → MouseLand/Kilosort

**This subproject uses conda, not uv** — this overrides the monorepo parent `CLAUDE.md`'s "always use uv" rule. Upstream Kilosort ships `environment.yml` + `setup.py` and expects conda for the CUDA/PyTorch toolchain; don't try to convert it.

## Commands

Install (editable, from a fresh conda env):
```bash
conda env create -f environment.yml
conda activate kilosort
pip install -e ".[gui]"
```

Run tests (from repo root):
```bash
pytest                          # fast unit tests only
pytest --runslow                # unit + slow regression tests (CPU)
pytest --gpu --runslow          # full suite on GPU (requires CUDA)
pytest tests/test_io.py         # single file
pytest tests/test_io.py::test_name    # single test
```

Run the GUI: `python -m kilosort`

Note: there is no linter or formatter configured — upstream does not use one.

## Architecture

Kilosort4 is a five-stage GPU spike sorter. The top-level driver is `kilosort/run_kilosort.py::_sort`; each stage mutates a shared `ops: dict` and hands off in-memory tensors. Understanding the `ops` flow is the key to productive work here.

1. **Preprocessing** (`compute_preprocessing` → `preprocessing.get_whitening_matrix`). Computes `ops['Wrot']` (whitening matrix) and `ops['fwav']` (highpass filter).
2. **Drift correction** (`compute_drift_correction` → `datashift.run`). Computes per-batch vertical shifts into `ops['dshift']`, shape `(Nbatches, nblocks)`. With `nblocks=0`, `dshift` is set to `None` and the drift branch in `io.py:BinaryFiltered.filter` is skipped.
3. **Spike detection** (`detect_spikes`). Two-stage: universal-template detection (`spikedetect.run`) then learned-template extraction via `template_matching.extract`. Produces `st0, tF, Wall`.
4. **Clustering** (`cluster_spikes` → `clustering_qr.run`, `template_matching.merging_function`). Graph-based QR clustering + CCG-based merge. Produces the final `(clu, Wall)`.
5. **Save** (`save_sorting` → `io.save_to_phy`). Writes phy-compatible output plus `ops.npy`.

Data flow is mediated by `kilosort/io.py::BinaryFiltered`: it wraps the raw binary, applies channel map / CAR / highpass / whitening / (optional) drift correction per batch. `tmin`/`tmax` restrict the time range; `ops['Nbatches']` is derived from the bfile.

Template matching (`kilosort/template_matching.py::extract`, `run_matching`) is basis-pursuit deconvolution: it takes a frozen template bank `U` and does NOT update it. The loop subtracts matched templates from the residual until no peak exceeds `Th_learned`. This is the primitive reused by `apply_templates`.

## Three distinct unit ID spaces (watch for this)

When reading/writing sorting outputs, distinguish:

- **KS cluster ID** (`spike_clusters.npy` values) — can have gaps from merges/removals.
- **Template index** (row in `templates.npy`, position in `Wall`) — dense, 0..N-1.
- **SI unit_id** / **spike vector unit_index** — SpikeInterface remapping; dense but not equal to KS cluster ID.

Indexing a dense positional array (`templates[ks_id]`, `Wall[ks_id]`) with a KS cluster ID is the classic bug — works below the first gap, silently wrong after.

## Fork-specific additions

This fork adds `kilosort.apply_templates` and `kilosort.apply_templates_chunked` (in `kilosort/run_kilosort.py`) — run only the matching-pursuit stage against a frozen `Wall` from a prior training run, for long recordings that can't be fully sorted. Requires training with `save_extra_vars=True`.

- Drift is inherited from training: `nblocks=0` skips drift everywhere; `nblocks>=1` recomputes `dshift` on the target recording (uses target's own average fingerprint as the reference, so training and target frames differ by ~zero for chronic probes within a session).
- `apply_templates_chunked` bounds peak host RAM by splitting the recording into time chunks (`chunk_hours` or explicit `chunk_ranges`), calling `apply_templates` per chunk, and merging phy outputs via `_merge_phy_chunks`. The merge uses mmap for `pc_features.npy`, runs cross-chunk `postprocessing.remove_duplicates`, and recomputes cluster tsvs from the merged spike train. Default is uniform re-extraction; `stitch_training=True` carves the training range out of the schedule and splices `train_out/` into the merge.

See `APPLY_TEMPLATES.md` at the repo root for the full workflow.

## Key files

- `kilosort/run_kilosort.py` — pipeline drivers, `ops` construction, logging, save. New public entry points go here.
- `kilosort/io.py` — `BinaryRWFile`, `BinaryFiltered`, `save_to_phy`, `save_ops`/`load_ops`. Any I/O or preprocessing change lives here.
- `kilosort/template_matching.py` — matching pursuit, template merging.
- `kilosort/clustering_qr.py` — graph-based clustering.
- `kilosort/datashift.py` — drift estimation.
- `kilosort/parameters.py` — `DEFAULT_SETTINGS` and parameter documentation.
- `tests/conftest.py` — defines `--gpu`, `--runslow`, `--download` CLI flags and common fixtures.
