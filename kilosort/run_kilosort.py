import time
from pathlib import Path
import logging
import shutil
import warnings
import platform
logger = logging.getLogger(__name__)

import numpy as np
import torch

import kilosort
from kilosort import (preprocessing, datashift, template_matching, clustering_qr,
                      clustering_qr, io, spikedetect, CCG, PROBE_DIR, postprocessing)
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.utils import (
    log_performance, log_cuda_details, probe_as_string, ops_as_string,
    get_performance, log_sorting_summary, log_thread_count
    )
import kilosort.plots as kplots

RECOGNIZED_SETTINGS = list(DEFAULT_SETTINGS.keys())
RECOGNIZED_SETTINGS.extend([
    'filename', 'data_dir', 'results_dir', 'probe_name', 'probe_path',
])
# These get mixed in with the other parameters when running through the GUI.
# When using the API, these should NOT be included in a settings dictionary
# even if they share a name with run_kilosort options.
GUI_SETTINGS = [
    'data_file_path', 'probe', 'data_dtype', 'save_preprocessed_copy',
    'clear_cache', 'do_CAR', 'invert_sign', 'verbose_log'
]


def run_kilosort(settings, probe=None, probe_name=None, filename=None,
                 data_dir=None, file_object=None, results_dir=None,
                 data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                 progress_bar=None, save_extra_vars=False, clear_cache=False,
                 save_preprocessed_copy=False, bad_channels=None, shank_idx=None,
                 verbose_console=False, verbose_log=False, torch_thread_lim=None):
    """Run full spike sorting pipeline on specified data.
    
    Parameters
    ----------
    settings : dict
        Specifies a number of configurable parameters used throughout the
        spike sorting pipeline. See `kilosort/parameters.py` for a full list of
        available parameters.
        NOTE: `n_chan_bin` must be specified here, but all other settings are
              optional.
    probe : dict; optional.
        A Kilosort4 probe dictionary, as returned by `kilosort.io.load_probe`.
    probe_name : str; optional.
        Filename of probe to use, within the default `PROBE_DIR`. Only include
        the filename without any preceeding directories. Will ony be used if
        `probe is None`. Alternatively, the full filepath to a probe stored in
        any directory can be specified with `settings = {'probe_path': ...}`.
        See `kilosort.utils` for default `PROBE_DIR` definition.
    filename: Path-like or list of Path-likes; optional.
        Full path to binary data file(s). If specified, will also set
        `data_dir = filename.parent`. If `filename` is a list, files will be
        treated as a single recording concatenated in time in the order provided.
    data_dir : str or Path; optional.
        Specifies directory where binary data file is stored. Kilosort will
        attempt to find the binary file. This works best if there is exactly one
        file in the directory with a .bin, .bat, .dat, or .raw extension.
        Only used if `filename is None`.
        Also see `kilosort.io.find_binary`.
    file_object : array-like file object; optional.
        Must have 'shape' and 'dtype' attributes and support array-like
        indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
        array or memmap. Must specify a valid `filename` as well, even though
        data will not be directly loaded from that file.
    results_dir : str or Path; optional.
        Directory where results will be stored. By default, will be set to
        `data_dir / 'kilosort4'`.
    data_dtype : str or type; optional.
        dtype of data in binary file, like `'int32'` or `np.uint16`. By default,
        dtype is assumed to be `'int16'`.
    do_CAR : bool; default=True.
        If True, apply common average reference during preprocessing
        (recommended).
    invert_sign : bool; default=False.
        If True, flip positive/negative values in data to conform to standard
        expected by Kilosort4.
    device : torch.device; optional.
        CPU or GPU device to use for PyTorch calculations. By default, PyTorch
        will use the first detected GPU. If no GPUs are detected, CPU will be
        used. To set this manually, specify `device = torch.device(<device_name>)`.
        See PyTorch documentation for full description.
    progress_bar : tqdm.std.tqdm or QtWidgets.QProgressBar; optional.
        Used by sorting steps and GUI to track sorting progress. Users should
        not need to specify this.
    save_extra_vars : bool; default=False.
        If True, save tF and Wall to disk after sorting.
    clear_cache : bool; default=False.
        If True, force pytorch to free up memory reserved for its cache in
        between memory-intensive operations.
        Note that setting `clear_cache=True` is NOT recommended unless you
        encounter GPU out-of-memory errors, since this can result in slower
        sorting.
    save_preprocessed_copy : bool; default=False.
        If True, save a pre-processed copy of the data (including drift
        correction) to `temp_wh.dat` in the results directory and format Phy
        output to use that copy of the data.
    bad_channels : list; optional.
        A list of channel indices (rows in the binary file) that should not be
        included in sorting. Listing channels here is equivalent to excluding
        them from the probe dictionary.
    shank_idx : float or list; optional.
        If not None, only channels from the specified shank index will be used.
        If a list is provided, each shank will be sorted sequentially and results
        will be saved in separate subfolders. Note that the shank_idx value(s)
        must match the actual value specified in `probe['kcoords']`. For example,
        `probe_idx=0` will not work if `probe['kcoords']` uses 1,2,3,4.
    verbose_console : bool; default=False.
        If True, set logging level for console output to `DEBUG` instead
        of `INFO`, so that additional information normally only saved to the
        log file will also show up in real time while sorting.
    verbose_log : bool; default=False.
        If True, include additional debug-level logging statements for some
        steps. This provides more detail for debugging, but may impact
        performance.
    torch_thread_lim : int; optional.
        If set, this will limit the number of pytorch threads on CPU.
        See docs for `torch.set_num_threads`.
    
    Raises
    ------
    ValueError
        If settings[`n_chan_bin`] is None (default). User must specify, for
        example:  `run_kilosort(settings={'n_chan_bin': 385})`.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st[:,0]`.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    similar_templates : np.ndarray.
        Similarity score between each pair of clusters, computed as correlation
        between clusters. Shape (n_clusters, n_clusters).
    is_ref : np.ndarray.
        1D boolean array with shape (n_clusters,) indicating whether each
        cluster is refractory.
    est_contam_rate : np.ndarray.
        Contamination rate for each cluster, computed as fraction of refractory
        period violations relative to expectation based on a Poisson process.
        Shape (n_clusters,).
    kept_spikes : np.ndarray.
        Boolean mask with shape (n_spikes,) that is False for spikes that were
        removed by `kilosort.postprocessing.remove_duplicate_spikes`
        and True otherwise.

    Notes
    -----
    For documentation of saved files, see `kilosort.io.save_to_phy`.

    """
    if torch_thread_lim is not None:
        torch.set_num_threads(torch_thread_lim)

    # Configure settings, ops, and file paths
    if settings is None or settings.get('n_chan_bin', None) is None:
        raise ValueError(
            '`n_chan_bin` is a required setting. This is the total number of '
            'channels in the binary file, which may or may not be equal to the '
            'number of channels specified by the probe.'
            )
    settings = {**DEFAULT_SETTINGS, **settings}
    # NOTE: This modifies settings in-place
    if not isinstance(shank_idx, list): shank_idx = [shank_idx]
    for idx in shank_idx:
        _filename, _data_dir, _results_dir, _probe = \
            set_files(settings, filename, probe, probe_name, data_dir,
                      results_dir, bad_channels, idx)
        setup_logger(_results_dir, verbose_console=verbose_console)

        ops, st, clu, tF, Wall, similar_templates, \
            is_ref, est_contam_rate, kept_spikes = _sort(
                _filename, _results_dir, _probe, settings, data_dtype, device,
                do_CAR, clear_cache, invert_sign, save_preprocessed_copy,
                verbose_log, save_extra_vars, file_object, progress_bar,
            )

    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes


def _sort(filename, results_dir, probe, settings, data_dtype, device, do_CAR,
          clear_cache, invert_sign, save_preprocessed_copy, verbose_log,
          save_extra_vars, file_object, progress_bar, gui_sorter=None):
    """Run sorting pipeline. See `run_kilosort` for documentation.
    
    Notes
    -----
    filename is expected to be a list of Paths at this point, even if it's
    a singleton list.
    
    """

    try:
        logger.info(f"Kilosort version {kilosort.__version__}")
        logger.info(f"Python version {platform.python_version()}")
        logger.info('-'*40)

        logger.info('System information:')
        logger.info(f'{platform.platform()} {platform.machine()}')
        logger.info(platform.processor())
        if device is None:
            if torch.cuda.is_available():
                logger.info('Using GPU for PyTorch computations. '
                            'Specify `device` to change this.')
                device = torch.device('cuda')
            else:
                logger.info('Using CPU for PyTorch computations. '
                            'Specify `device` to change this.')
                device = torch.device('cpu')

        if device != torch.device('cpu'):
            memory = torch.cuda.get_device_properties(device).total_memory/1024**3
            logger.info(f'Using CUDA device: {torch.cuda.get_device_name()} {memory:.2f}GB')

        logger.info('-'*40)
        if len(filename) == 1:
            logger.info(f"Sorting {filename}")
        else:
            logger.info(f"Sorting {filename[0].parent}/... (multiple files)")

        if data_dtype is None:
            logger.info(
                "Interpreting binary file as default dtype='int16'. If data was "
                "saved in a different format, specify `data_dtype`."
                )
            data_dtype = 'int16'

        if not do_CAR:
            logger.info("Skipping common average reference.")
        if clear_cache:
            logger.info('clear_cache=True')

        if probe['chanMap'].max() >= settings['n_chan_bin']:
            raise ValueError(
                f'Largest value of chanMap exceeds channel count of data, '
                'make sure chanMap is 0-indexed.'
            )

        tic0 = time.time()
        ops, settings = initialize_ops(
            settings, probe, data_dtype, do_CAR, invert_sign,
            device, save_preprocessed_copy, gui_mode=(gui_sorter is not None)
            )
        
        # Pretty-print ops and probe for log
        logger.debug(f"Initial ops:\n\n{ops_as_string(ops)}\n")
        logger.debug(f"Probe dictionary:\n\n{probe_as_string(ops['probe'])}\n")

        # Baseline performance metrics
        log_performance(logger, 'info', 'Resource usage before sorting')
        log_thread_count(logger)

        # Set preprocessing and drift correction parameters
        ops = compute_preprocessing(ops, device, tic0=tic0, file_object=file_object)
        np.random.seed(1)
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1) 
        ops, bfile, st0 = compute_drift_correction(
            ops, device, tic0=tic0, progress_bar=progress_bar,
            file_object=file_object, clear_cache=clear_cache,
            verbose=verbose_log
            )

        log_thread_count(logger)

        # Save preprocessing steps
        if save_preprocessed_copy:
            io.save_preprocessing(results_dir / 'temp_wh.dat', ops, bfile)
            log_performance(logger, 'info', 'Resource usage after saving preprocessing.',
                            reset=True)

        logger.info('Generating drift plots ...')
        # st0 will be None if nblocks = 0 (no drift correction)
        if st0 is not None:
            if gui_sorter is not None:
                gui_sorter.dshift = ops['dshift']
                gui_sorter.st0 = st0
                gui_sorter.plotDataReady.emit('drift')
            else:
                kplots.plot_drift_amount(ops, results_dir, tmin=settings['tmin'])
                kplots.plot_drift_scatter(st0, results_dir, tmin=settings['tmin'])

        # Sort spikes and save results
        st,tF, Wall0, clu0 = detect_spikes(
            ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
            clear_cache=clear_cache, verbose=verbose_log
            )

        log_thread_count(logger)

        logger.info('Generating diagnostic plots ...')
        if gui_sorter is not None:
            gui_sorter.Wall0 = Wall0
            gui_sorter.wPCA = torch.clone(ops['wPCA'].cpu()).numpy()
            gui_sorter.clu0 = clu0
            gui_sorter.plotDataReady.emit('diagnostics')
        else:
            kplots.plot_diagnostics(Wall0, clu0, ops, results_dir)

        clu, Wall, st, tF = cluster_spikes(
            st, tF, ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
            clear_cache=clear_cache, verbose=verbose_log,
            )

        log_thread_count(logger)

        ops, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            save_sorting(
                ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0,
                save_extra_vars=save_extra_vars,
                save_preprocessed_copy=save_preprocessed_copy,
                skip_dat_path=(file_object is not None)
                )
        if torch.cuda.is_available():
            ops['cuda_postproc'] = torch.cuda.memory_stats(device)

        log_thread_count(logger)

        logger.info('Generating spike position plot ...')
        if gui_sorter is not None:
            gui_sorter.clu = clu[kept_spikes]
            gui_sorter.is_refractory = is_ref
            gui_sorter.plotDataReady.emit('probe')
        else:
            kplots.plot_spike_positions(clu[kept_spikes], is_ref, results_dir)
        logger.info('Sorting finished.')
        log_sorting_summary(ops, log=logger, level='info')
        
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            logger.exception('Out of memory error, printing performance...')
            log_cuda_details(logger)
            log_performance(logger, level='info')

        # This makes sure the full traceback is written to log file.
        logger.exception('Encountered error in `run_kilosort`:')
        # Annoyingly, this will print the error message twice for console, but
        # I haven't found a good way around that.
        raise
    
    finally:
        close_logger()


    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes


def apply_templates(training_dir, filename, results_dir=None, tmin=0.0,
                    tmax=np.inf, device=None, file_object=None,
                    progress_bar=None, verbose_console=False,
                    torch_thread_lim=None):
    """Extract spikes from a recording using frozen templates from a prior run.

    Runs only the basis-pursuit (matching-pursuit) stage of Kilosort4 against
    a fixed template bank `Wall` loaded from a previous `run_kilosort` run.
    Skips universal-template spike detection, clustering, and merging, so the
    final cluster IDs are identical to those produced during training.

    If the training run used drift correction (`nblocks >= 1`), drift is
    re-estimated on the target recording (independently of the training
    drift) and applied before template matching. This produces a per-batch
    `dshift` sized to the target bfile, using the target recording's own
    average fingerprint as the reference. For chronic probes with small
    drift, the difference between the training reference frame and the
    target reference frame is typically well below `binning_depth`, so
    templates still match.

    This is intended for very long recordings where a full sort is infeasible
    but a short training window is representative.

    Requires the training run to have been performed with `save_extra_vars=True`
    so that `Wall.npy` exists alongside `ops.npy`.

    Parameters
    ----------
    training_dir : str or Path
        Directory containing `ops.npy` and `Wall.npy` from a prior KS4 run.
    filename : Path-like or list of Path-likes
        Binary file(s) to extract from. Must match the training recording in
        channel count, sampling rate, dtype, and probe geometry.
    results_dir : str or Path; optional.
        Output directory. Defaults to `filename.parent / 'kilosort4_apply'`.
    tmin, tmax : float; optional.
        Time range (seconds) within `filename` to extract. Defaults to the
        full recording.
    device : torch.device; optional.
        Defaults to CUDA if available, else CPU.
    file_object : array-like; optional.
        Alternative to reading from disk; see `run_kilosort`.
    progress_bar : optional.
        Forwarded to `template_matching.extract`.
    verbose_console : bool; optional.
        If True, set console logging level to DEBUG.
    torch_thread_lim : int; optional.
        If set, limits pytorch CPU thread count.

    Returns
    -------
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate,
    kept_spikes : same shapes/types as `run_kilosort`.

    Notes
    -----
    Memory: `tF` is held in RAM for the full recording before `save_to_phy`.
    A 24-hr recording with ~200 units firing at ~10 Hz and default
    `nearest_chans`/`n_pcs` needs ~40 GB of host memory for `tF`. If this is
    a problem, reduce the time range or run in segments.
    """
    if torch_thread_lim is not None:
        torch.set_num_threads(torch_thread_lim)

    training_dir = Path(training_dir)
    if not isinstance(filename, list):
        filename = [filename]
    filename = [Path(f) for f in filename]
    for f in filename:
        if not f.exists():
            raise FileExistsError(f"filename '{f}' does not exist")

    if results_dir is None:
        results_dir = filename[0].parent / 'kilosort4_apply'
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(results_dir, verbose_console=verbose_console)
    try:
        tic0 = time.time()
        logger.info(f"Kilosort version {kilosort.__version__}")
        logger.info(f"Python version {platform.python_version()}")
        logger.info('-'*40)
        logger.info(f'apply_templates: training_dir={training_dir}')
        logger.info(f'apply_templates: target filename={filename}')
        logger.info(f'apply_templates: tmin={tmin}, tmax={tmax}')

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        logger.info(f'Using device: {device}')

        ops_path = training_dir / 'ops.npy'
        wall_path = training_dir / 'Wall.npy'
        if not ops_path.exists():
            raise FileNotFoundError(f'ops.npy not found in {training_dir}')
        if not wall_path.exists():
            raise FileNotFoundError(
                f'Wall.npy not found in {training_dir}. '
                'Re-run training with save_extra_vars=True.'
            )

        ops = io.load_ops(ops_path, device=device)
        Wall = torch.from_numpy(np.load(wall_path)).to(device)
        logger.info(f'Loaded frozen templates: Wall shape {tuple(Wall.shape)}')

        # Discard training's dshift; it's indexed by training batches and
        # would be wrong to apply to the target bfile. If the training run
        # used drift (nblocks >= 1), we recompute dshift on the target bfile
        # below.
        ops['dshift'] = None
        nblocks = int(ops.get('nblocks', 0))

        # Override filename and time range; everything else stays frozen.
        ops['filename'] = filename
        ops['data_dir'] = filename[0].parent
        ops['settings']['filename'] = filename
        ops['settings']['data_dir'] = filename[0].parent
        ops['settings']['tmin'] = float(tmin)
        ops['settings']['tmax'] = float(tmax)
        ops['tmin'] = float(tmin)
        ops['tmax'] = float(tmax)

        n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
            _, _, _, _, artifact, shift, scale, batch_ds = \
                get_run_parameters(ops)

        bfile = io.BinaryFiltered(
            ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map,
            hp_filter=ops['fwav'], whiten_mat=ops['Wrot'], dshift=None,
            device=device, do_CAR=do_CAR, invert_sign=invert, dtype=dtype,
            tmin=tmin, tmax=tmax, artifact_threshold=artifact,
            shift=shift, scale=scale, file_object=file_object,
            batch_downsampling=batch_ds,
        )
        ops['Nbatches'] = bfile.n_batches
        logger.info(f'N samples: {bfile.n_samples}')
        logger.info(f'N seconds: {bfile.n_samples/fs}')
        logger.info(f'N batches: {bfile.n_batches}')

        if nblocks >= 1:
            logger.info(' ')
            logger.info(f'Recomputing drift on target recording (nblocks={nblocks})')
            logger.info('-'*40)
            tic = time.time()
            ops, st0 = datashift.run(
                ops, bfile, device=device, progress_bar=progress_bar,
            )
            logger.info(f'drift recomputed in {time.time()-tic:.2f}s')
            if st0 is not None:
                kplots.plot_drift_amount(ops, results_dir, tmin=float(tmin))
                kplots.plot_drift_scatter(st0, results_dir, tmin=float(tmin))
            bfile = io.BinaryFiltered(
                ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map,
                hp_filter=ops['fwav'], whiten_mat=ops['Wrot'],
                dshift=ops['dshift'], device=device, do_CAR=do_CAR,
                invert_sign=invert, dtype=dtype, tmin=tmin, tmax=tmax,
                artifact_threshold=artifact, shift=shift, scale=scale,
                file_object=file_object, batch_downsampling=batch_ds,
            )

        logger.info(' ')
        logger.info('Extracting spikes with frozen templates')
        logger.info('-'*40)
        tic = time.time()
        st, tF, ops = template_matching.extract(
            ops, bfile, Wall, device=device, progress_bar=progress_bar,
        )
        logger.info(f'{len(st)} spikes extracted in {time.time()-tic:.2f}s')
        clu = st[:, 1].astype('int32')

        ops, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            save_sorting(
                ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0=tic0,
                save_extra_vars=False, save_preprocessed_copy=False,
                skip_dat_path=(file_object is not None),
            )
    except Exception:
        logger.exception('Encountered error in `apply_templates`:')
        raise
    finally:
        close_logger()

    return ops, st, clu, tF, Wall, similar_templates, \
           is_ref, est_contam_rate, kept_spikes


def apply_templates_chunked(training_dir, filename, chunk_hours=4.0,
                            chunk_ranges=None, results_dir=None,
                            tmin=0.0, tmax=np.inf, stitch_training=False,
                            keep_chunks=False, device=None, file_object=None,
                            progress_bar=None, verbose_console=False,
                            torch_thread_lim=None):
    """Run `apply_templates` in chunks and merge the phy outputs.

    Splits the target recording into time chunks, runs `apply_templates` on
    each chunk to a temporary phy directory, then merges the per-chunk phy
    outputs into a single phy directory at `results_dir`.

    Peak host RAM is bounded by a single chunk's `tF` (plus the merge step,
    which mmaps the per-chunk `pc_features.npy` files).

    Parameters
    ----------
    training_dir : str or Path
        Directory containing `ops.npy` and `Wall.npy` from a prior KS4 run.
    filename : Path-like or list of Path-likes
        Target binary file(s). A list is treated as one virtual recording
        concatenated in the given order, independent of chunk boundaries.
    chunk_hours : float; default=4.0
        Chunk duration in hours when `chunk_ranges` is None.
    chunk_ranges : list of (float, float); optional
        Explicit chunk boundaries as (t_start, t_end) pairs in seconds.
        Overrides `chunk_hours`. Ranges must be disjoint and in increasing
        order. Useful for aligning chunks with physical file boundaries.
    results_dir : str or Path; optional
        Output directory. Defaults to `filename.parent / 'kilosort4_apply'`.
    tmin, tmax : float; optional
        Overall time range to cover across chunks.
    stitch_training : bool; default=False
        If True, include the training window (loaded from
        `training_dir/ops.npy`'s tmin/tmax) as one of the inputs to the
        merge, and skip any chunk that overlaps it. Use this only if you
        want to preserve training's saved spike train; note that training
        spikes were extracted with first-pass templates while apply chunks
        use the frozen final templates, so the merged output will have a
        small methodological discontinuity at the train/apply boundary.
    keep_chunks : bool; default=False
        If True, leave the per-chunk phy directories in place after the
        merge (named `_chunk_0000/`, etc. under `results_dir`).
    device, file_object, progress_bar, verbose_console, torch_thread_lim
        Forwarded to each chunk's `apply_templates` call.

    Returns
    -------
    results_dir : Path
        Path to the merged phy output directory.

    Notes
    -----
    Each chunk writes a full phy directory via `apply_templates` (which
    calls `save_to_phy`). The merge step concatenates the per-spike
    arrays (already sorted within each chunk and chunks are time-disjoint,
    so the concatenation is already time-sorted), runs a final
    cross-chunk `remove_duplicates` on the merged spike train, copies the
    cluster-level / channel-level / whitening files from the first chunk
    (they're identical across chunks because `Wall` is frozen), and
    recomputes `cluster_KSLabel.tsv`/`cluster_ContamPct.tsv` from the
    merged train via `CCG.refract`.
    """
    if torch_thread_lim is not None:
        torch.set_num_threads(torch_thread_lim)

    training_dir = Path(training_dir)
    if not isinstance(filename, list):
        filename = [filename]
    filename = [Path(f) for f in filename]
    for f in filename:
        if not f.exists():
            raise FileExistsError(f"filename '{f}' does not exist")
    if results_dir is None:
        results_dir = filename[0].parent / 'kilosort4_apply'
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Peek at training ops to get fs, n_chan_bin, dtype, and training range.
    ops_peek = io.load_ops(training_dir / 'ops.npy', device=torch.device('cpu'))
    fs = float(ops_peek['fs'])
    n_chan_bin = int(ops_peek['n_chan_bin'])
    data_dtype = ops_peek['data_dtype']

    # Compute recording duration honoring file_object if provided.
    if file_object is not None:
        total_samples = file_object.shape[0]
    elif len(filename) > 1:
        total_samples = sum(
            io.get_total_samples(f, n_chan_bin, data_dtype) for f in filename
        )
    else:
        total_samples = io.get_total_samples(filename[0], n_chan_bin, data_dtype)
    total_sec = total_samples / fs
    effective_tmin = max(0.0, float(tmin))
    effective_tmax = total_sec if not np.isfinite(tmax) else min(float(tmax), total_sec)
    if effective_tmin >= effective_tmax:
        raise ValueError(
            f'Empty time range: tmin={effective_tmin} >= tmax={effective_tmax}'
        )

    # Resolve training range if stitching.
    train_range = None
    if stitch_training:
        train_tmin = float(ops_peek.get('tmin', 0.0))
        train_tmax_raw = ops_peek.get('tmax', np.inf)
        train_tmax = (total_sec if not np.isfinite(train_tmax_raw)
                      else float(train_tmax_raw))
        train_range = (train_tmin, train_tmax)

    # Build chunk schedule.
    if chunk_ranges is None:
        if chunk_hours is None or chunk_hours <= 0:
            raise ValueError('Either `chunk_hours > 0` or `chunk_ranges` must be given.')
        chunk_sec = float(chunk_hours) * 3600.0
        ranges = []
        t = effective_tmin
        while t < effective_tmax:
            ranges.append((t, min(t + chunk_sec, effective_tmax)))
            t += chunk_sec
    else:
        # Validate and clamp user-supplied ranges.
        ranges = []
        prev_end = effective_tmin
        for t0, t1 in chunk_ranges:
            t0, t1 = float(t0), float(t1)
            if t0 < prev_end or t1 <= t0 or t1 > effective_tmax:
                raise ValueError(
                    f'Invalid chunk_ranges: ({t0}, {t1}) overlaps or exceeds '
                    f'overall tmin={effective_tmin}, tmax={effective_tmax}.'
                )
            ranges.append((t0, t1))
            prev_end = t1

    # If stitching, carve training window out of the chunk schedule.
    apply_ranges = ranges
    if train_range is not None:
        apply_ranges = _subtract_range(ranges, train_range)
        logger.info(
            f'apply_templates_chunked: stitching training window '
            f'[{train_range[0]:.1f}, {train_range[1]:.1f}]; '
            f'{len(apply_ranges)} apply chunks after carve-out'
        )

    logger.info(
        f'apply_templates_chunked: {len(apply_ranges)} apply chunk(s), '
        f'chunk_hours={chunk_hours}, overall [{effective_tmin:.1f}, '
        f'{effective_tmax:.1f}] s'
    )

    chunk_dirs = []
    for k, (t0, t1) in enumerate(apply_ranges):
        chunk_dir = results_dir / f'_chunk_{k:04d}'
        logger.info(
            f'Chunk {k+1}/{len(apply_ranges)}: tmin={t0:.1f} tmax={t1:.1f}'
        )
        apply_templates(
            training_dir, filename, results_dir=chunk_dir,
            tmin=t0, tmax=t1, device=device, file_object=file_object,
            progress_bar=progress_bar, verbose_console=verbose_console,
            torch_thread_lim=torch_thread_lim,
        )
        chunk_dirs.append(chunk_dir)

    # Merge inputs: apply chunks plus optionally training_dir.
    merge_inputs = list(chunk_dirs)
    if train_range is not None:
        merge_inputs.append(training_dir)
    _merge_phy_chunks(
        merge_inputs, results_dir,
        filename=filename, effective_tmin=effective_tmin,
        effective_tmax=effective_tmax,
    )

    if not keep_chunks:
        for d in chunk_dirs:
            shutil.rmtree(d, ignore_errors=True)

    logger.info(f'Merged phy output at {results_dir}')
    return results_dir


def _subtract_range(ranges, cut):
    """Remove `cut = (c0, c1)` from an ordered list of disjoint `ranges`."""
    c0, c1 = cut
    out = []
    for t0, t1 in ranges:
        if t1 <= c0 or t0 >= c1:
            out.append((t0, t1))
            continue
        if t0 < c0:
            out.append((t0, c0))
        if t1 > c1:
            out.append((c1, t1))
    return out


def _merge_phy_chunks(input_dirs, out_dir, filename, effective_tmin,
                      effective_tmax):
    """Merge per-chunk phy directories into a single phy directory.

    Assumes each `input_dir` is a complete phy output (as produced by
    `apply_templates` or a prior KS4 run) and that their time ranges are
    disjoint. Inputs are sorted by the minimum spike time found in each.
    """
    input_dirs = [Path(d) for d in input_dirs]

    # Sort inputs by first spike time (equivalently by tmin of the chunk).
    def _first_time(d):
        st = np.load(d / 'spike_times.npy', mmap_mode='r')
        return int(st[0]) if len(st) else np.iinfo(np.int64).max
    input_dirs = sorted(input_dirs, key=_first_time)

    per_spike_1d = [
        'spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy',
        'spike_detection_templates.npy', 'amplitudes.npy', 'kept_spikes.npy',
    ]
    # spike_positions is (n_spikes, 2); handled separately.

    # Concatenate the 1D per-spike arrays (already time-sorted within each
    # chunk; chunks are time-disjoint so concatenation is sorted).
    merged = {}
    for name in per_spike_1d:
        arrs = [np.load(d / name) for d in input_dirs]
        merged[name] = np.concatenate(arrs, axis=0)
    sp_arrs = [np.load(d / 'spike_positions.npy') for d in input_dirs]
    merged['spike_positions.npy'] = np.concatenate(sp_arrs, axis=0)

    # Cross-chunk dedup on the merged train.
    ops = io.load_ops(input_dirs[0] / 'ops.npy', device=torch.device('cpu'))
    dt = int(ops.get('duplicate_spike_bins', 15))
    spike_times, spike_clusters, kept = postprocessing.remove_duplicates(
        merged['spike_times.npy'], merged['spike_clusters.npy'], dt=dt,
    )
    logger.info(
        f'_merge_phy_chunks: {len(spike_times)} spikes after cross-chunk '
        f'dedup (removed {int((~kept).sum())})'
    )

    # Apply kept-mask to all per-spike arrays.
    for name in per_spike_1d:
        if name == 'spike_times.npy':
            merged[name] = spike_times
        elif name == 'spike_clusters.npy':
            merged[name] = spike_clusters
        elif name == 'kept_spikes.npy':
            # Merged output: everything that survives dedup is "kept".
            merged[name] = np.ones(len(spike_times), dtype=bool)
        else:
            merged[name] = merged[name][kept]
    merged['spike_positions.npy'] = merged['spike_positions.npy'][kept]

    # Save the merged 1D/2D per-spike arrays.
    for name, arr in merged.items():
        np.save(out_dir / name, arr)

    # Merge pc_features.npy via a mmap-backed concatenation to avoid
    # loading everything into RAM, then apply the kept-mask.
    pc_shapes = [np.load(d / 'pc_features.npy', mmap_mode='r').shape
                 for d in input_dirs]
    total_pre = sum(s[0] for s in pc_shapes)
    tail_shape = pc_shapes[0][1:]
    pc_dtype = np.load(input_dirs[0] / 'pc_features.npy', mmap_mode='r').dtype
    tmp_pc = out_dir / '_pc_features_concat.npy'
    pc_cat = np.lib.format.open_memmap(
        tmp_pc, mode='w+', dtype=pc_dtype,
        shape=(total_pre,) + tail_shape,
    )
    offset = 0
    for d, s in zip(input_dirs, pc_shapes):
        chunk_pc = np.load(d / 'pc_features.npy', mmap_mode='r')
        n = s[0]
        pc_cat[offset:offset + n] = chunk_pc[:]
        offset += n
    pc_cat.flush()
    del pc_cat
    pc_cat = np.load(tmp_pc, mmap_mode='r')
    # kept is over the pre-dedup order, same order as pc_cat. Write the
    # final pc_features.npy in normal (non-mmap) form.
    np.save(out_dir / 'pc_features.npy', np.asarray(pc_cat[kept]))
    del pc_cat
    tmp_pc.unlink()

    # Copy cluster-level / channel-level / whitening files from first input.
    # These are identical across chunks because Wall and the probe are fixed.
    # For `stitch_training`, templates.npy (and the similarity matrix) from
    # a regular apply chunk matches train_out's too because both were
    # produced from the same frozen Wall.
    copy_files = [
        'templates.npy', 'templates_ind.npy', 'similar_templates.npy',
        'pc_feature_ind.npy', 'channel_map.npy', 'channel_positions.npy',
        'channel_shanks.npy', 'whitening_mat.npy', 'whitening_mat_inv.npy',
        'whitening_mat_dat.npy',
    ]
    for f in copy_files:
        src = input_dirs[0] / f
        if src.exists():
            shutil.copy(src, out_dir / f)

    # Recompute cluster-level tsvs from merged spike train.
    acg_t = ops['settings']['acg_threshold']
    ccg_t = ops['settings']['ccg_threshold']
    is_ref, est_contam = CCG.refract(
        spike_clusters, spike_times / ops['fs'],
        acg_threshold=acg_t, ccg_threshold=ccg_t,
    )
    # Template amplitudes (for cluster_Amplitude.tsv) can be read from
    # any input's template_amplitudes via templates.npy norms.
    templates = np.load(out_dir / 'templates.npy')
    template_amplitudes = np.linalg.norm(templates, axis=(1, 2))
    stypes = ['ContamPct', 'Amplitude', 'KSLabel']
    ks_labels = [['mua', 'good'][int(r)] for r in is_ref]
    props = [est_contam * 100, template_amplitudes, ks_labels]
    for stype, prop in zip(stypes, props):
        with open(out_dir / f'cluster_{stype}.tsv', 'w') as fh:
            fh.write(f'cluster_id\t{stype}\n')
            for i, p in enumerate(prop):
                if stype != 'KSLabel':
                    fh.write(f'{i}\t{p:.1f}\n')
                else:
                    fh.write(f'{i}\t{p}\n')
    shutil.copyfile(out_dir / 'cluster_KSLabel.tsv',
                    out_dir / 'cluster_group.tsv')

    # Write params.py pointing at the full target filename list.
    dtype = f"'{ops['data_dtype']}'"
    dat_path = [Path(p).resolve().as_posix() for p in filename]
    params = {
        'n_channels_dat': ops['settings']['n_chan_bin'],
        'offset': 0,
        'sample_rate': ops['settings']['fs'],
        'dtype': dtype,
        'hp_filtered': False,
        'dat_path': dat_path,
    }
    with open(out_dir / 'params.py', 'w') as fh:
        for key in params:
            fh.write(f'{key} = {params[key]}\n')

    # Save a merged ops.npy with the full time range.
    ops['filename'] = [Path(f) for f in filename]
    ops['data_dir'] = Path(filename[0]).parent
    ops['settings']['filename'] = ops['filename']
    ops['settings']['data_dir'] = ops['data_dir']
    ops['tmin'] = float(effective_tmin)
    ops['tmax'] = float(effective_tmax)
    ops['settings']['tmin'] = ops['tmin']
    ops['settings']['tmax'] = ops['tmax']
    ops['n_spikes'] = int(len(spike_times))
    ops['n_units_total'] = int(len(is_ref))
    ops['n_units_good'] = int(is_ref.sum())
    io.save_ops(ops, out_dir)


def set_files(settings, filename, probe, probe_name, data_dir, results_dir,
              bad_channels, shank_idx):
    """Parse file and directory information for data, probe, and results."""

    # Check for filename 
    filename = settings.get('filename', None) if filename is None else filename 

    # Use data_dir if filename not available
    if filename is None:
        data_dir = settings.get('data_dir', None) if data_dir is None else data_dir
        if data_dir is None:
            raise ValueError('no path to data provided, set "data_dir=" or "filename="')
        data_dir = Path(data_dir).resolve()
        if not data_dir.exists():
            raise FileExistsError(f"data_dir '{data_dir}' does not exist")

        # Find binary file in the folder
        filename  = io.find_binary(data_dir=data_dir)
        filename = [filename]
    else:
        if not isinstance(filename, list):
            filename = [filename]
        filename = [Path(f) for f in filename]
        for f in filename:
            if not f.exists():
                raise FileExistsError(f"filename '{filename}' does not exist")
        data_dir = filename[0].parent
        
    # Convert paths to strings when saving to ops, otherwise ops can only
    # be loaded on the operating system that originally ran the code.
    settings['filename'] = filename
    settings['data_dir'] = data_dir

    # Try to set results_dir based on settings, otherwise use default.
    results_dir = settings.get('results_dir', None) if results_dir is None else results_dir
    results_dir = Path(results_dir).resolve() if results_dir is not None else None
    if results_dir is None:
        results_dir = data_dir / 'kilosort4'
    if shank_idx is not None:
        results_dir = results_dir / f'shank_{shank_idx}'
    # Make sure results directory exists
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # find probe configuration file and load
    if probe is None:
        if probe_name is not None:     probe_path = PROBE_DIR / probe_name
        elif 'probe_name' in settings: probe_path = PROBE_DIR / settings['probe_name']
        elif 'probe_path' in settings: probe_path = Path(settings['probe_path']).resolve()
        else: raise ValueError('no probe_name or probe_path provided, set probe_name=')
        if not probe_path.exists():
            raise FileExistsError(f"probe_path '{probe_path}' does not exist")
        
        probe  = io.load_probe(probe_path)
    else:
        # Make sure xc, yc are float32, otherwise there are casting problems
        # with some pytorch functions.
        probe['xc'] = probe['xc'].astype(np.float32)
        probe['yc'] = probe['yc'].astype(np.float32)

    # Let user know if there are too many dimensions in probe entries.
    # Don't want to automatically flatten them incase they've made assumptions
    # about higher-D ordering.
    for k in ['xc', 'yc', 'kcoords', 'chanMap']:
        if probe[k].ndim > 1:
            raise ValueError(f"Array-valued probe entries should have 1 dim, "
                             f"but key: {k} has ndim == {probe[k].ndim}.")

    if bad_channels is not None:
        probe = io.remove_bad_channels(probe, bad_channels)
    if shank_idx is not None:
        probe = io.select_shank(probe, shank_idx)

    return filename, data_dir, results_dir, probe


def setup_logger(results_dir, verbose_console=False):
    results_dir = Path(results_dir)
    
    # Get root logger for Kilosort application
    ks_log = logging.getLogger('kilosort')
    ks_log.setLevel(logging.DEBUG)

    # Add file handler at debug level, include timestamps and logging level
    # in text output.
    file = logging.FileHandler(results_dir / 'kilosort4.log', mode='w')
    file.setLevel(logging.DEBUG)
    text_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    file_formatter = logging.Formatter(text_format)
    file.setFormatter(file_formatter)

    # Skip this if the handlers were already added, like when running multiple
    # times in a single session.
    if not ks_log.handlers:
        # Add console handler at info level with shorter messages,
        # unless verbose is requested.
        console = logging.StreamHandler()
        if verbose_console:
            console.setLevel(logging.DEBUG)
            console.setFormatter(file_formatter)
        else:
            console.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(name)-12s: %(message)s')
            console.setFormatter(console_formatter)
        ks_log.addHandler(console)

    # Always add file handler since log file might change locations
    ks_log.addHandler(file)


def close_logger():
    ks_log = logging.getLogger('kilosort')
    for handler in ks_log.handlers.copy():
        ks_log.removeHandler(handler)
        handler.close()


def initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign,
                   device, save_preprocessed_copy, gui_mode=False) -> dict:
    """Package settings and probe information into a single `ops` dictionary."""

    settings = settings.copy()
    if settings['nt'] % 2 == 0:
        raise ValueError(f'`nt` must be odd, but got nt={settings["nt"]}')
    if settings['nt0min'] is None:
        settings['nt0min'] = int(20 * settings['nt']/61)
    if settings['max_channel_distance'] is None:
        # Default used to be None, now it's a constant. Adding this so that
        # cached settings values in the GUI don't cause disruption.
        settings['max_channel_distance'] = DEFAULT_SETTINGS['max_channel_distance']

    if settings['nearest_chans'] > len(probe['chanMap']):
        msg = f"""
            Parameter `nearest_chans` must be less than or equal to the number 
            of data channels being sorted.\n
            Changing from {settings['nearest_chans']} to {len(probe['chanMap'])}.
            """
        warnings.warn(msg, UserWarning)
        settings['nearest_chans'] = len(probe['chanMap'])

    if 'duplicate_spike_bins' in settings:
        msg = """
            The `duplicate_spike_bins` parameter has been replaced with 
            `duplicate_spike_ms`. Specifying the former will have no effect, 
            since it gets overwritten based on sampling rate.
            """
        warnings.warn(msg, DeprecationWarning)
    dup_bins = int(settings['duplicate_spike_ms'] * (settings['fs']/1000))

    # If running through GUI, also allow some additional relevant keys in
    # settings dictionary.
    recognized = RECOGNIZED_SETTINGS.copy()
    if gui_mode:
        recognized.extend(GUI_SETTINGS.copy())

    # Raise an error if there are unrecognized settings entries to make users
    # aware if they've made a typo, are using a deprecated setting, etc.
    unrecognized = []
    for k, _ in settings.items():
        if k not in recognized:
            unrecognized.append(k)
    if len(unrecognized) > 0:
        logger.info('Unrecognized keys found in `settings`')
        logger.info('See `kilosort.run_kilosort.RECOGNIZED_SETTINGS`')
        raise ValueError(f'Unrecognized settings: {unrecognized}')


    # TODO: Clean this up during refactor. Lots of confusing duplication here.
    ops = settings.copy()
    ops['settings'] = settings
    ops['probe'] = probe
    ops['data_dtype'] = data_dtype
    ops['do_CAR'] = do_CAR
    ops['invert_sign'] = invert_sign
    ops['NTbuff'] = ops['batch_size'] + 2 * ops['nt']
    ops['Nchan'] = len(probe['chanMap'])
    ops['n_chan_bin'] = settings['n_chan_bin']
    ops['duplicate_spike_bins'] = dup_bins
    ops['torch_device'] = str(device)
    ops['save_preprocessed_copy'] = save_preprocessed_copy

    if not settings['templates_from_data'] and settings['nt'] != 61:
        raise ValueError('If using pre-computed universal templates '
                         '(templates_from_data=False), nt must be 61')

    ops = {**ops, **probe}

    return ops, settings

def get_run_parameters(ops) -> list:
    """Get `ops` dict values needed by `run_kilosort` subroutines."""

    parameters = [
        ops['settings']['n_chan_bin'],
        ops['settings']['fs'],
        ops['settings']['batch_size'],
        ops['settings']['nt'],
        ops['settings']['nt0min'],  # also called twav_min
        ops['probe']['chanMap'],
        ops['data_dtype'],
        ops['do_CAR'],
        ops['invert_sign'],
        ops['probe']['xc'],
        ops['probe']['yc'],
        ops['settings']['tmin'],
        ops['settings']['tmax'],
        ops['settings']['artifact_threshold'],
        ops['settings']['shift'],
        ops['settings']['scale'],
        ops['settings']['batch_downsampling']
    ]

    return parameters


def compute_preprocessing(ops, device, tic0=np.nan, file_object=None):
    """Compute preprocessing parameters and save them to `ops`.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    tic0 : float; default=np.nan
        Start time of `run_kilosort`.
    file_object : array-like file object; optional.
        Must have 'shape' and 'dtype' attributes and support array-like
        indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
        array or memmap.

    Returns
    -------
    ops : dict
    
    """

    tic = time.time()
    logger.info(' ')
    logger.info('Computing preprocessing variables.')
    logger.info('-'*40)

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        xc, yc, tmin, tmax, artifact, shift, scale, batch_downsampling = \
            get_run_parameters(ops)
    nskip = ops['settings']['nskip']
    whitening_range = ops['settings']['whitening_range']
    
    # Compute high pass filter
    cutoff = ops['settings']['highpass_cutoff']
    hp_filter = preprocessing.get_highpass_filter(fs, cutoff, device=device)
    # Compute whitening matrix
    bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min,
                              chan_map, hp_filter, device=device, do_CAR=do_CAR,
                              invert_sign=invert, dtype=dtype, tmin=tmin,
                              tmax=tmax, artifact_threshold=artifact,
                              shift=shift, scale=scale, file_object=file_object,
                              batch_downsampling=batch_downsampling)

    logger.info(f'N samples: {bfile.n_samples}')
    logger.info(f'N seconds: {bfile.n_samples/fs}')
    logger.info(f'N batches: {bfile.n_batches}')

    whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip,
                                                    nrange=whitening_range)


    # Save results
    ops['Nbatches'] = bfile.n_batches
    ops['preprocessing'] = {}
    ops['preprocessing']['whiten_mat'] = whiten_mat
    ops['preprocessing']['hp_filter'] = hp_filter
    ops['Wrot'] = whiten_mat
    ops['fwav'] = hp_filter

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_preproc'] = elapsed
    ops['usage_preproc'] = get_performance()
    logger.info(f'Preprocessing filters computed in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'hp_filter shape: {hp_filter.shape}')
    logger.debug(f'whiten_mat shape: {whiten_mat.shape}')
    # Check scale of data for log file
    b1 = bfile.padded_batch_to_torch(0).cpu().numpy()
    logger.debug(f"First batch min, max: {b1.min(), b1.max()}")

    log_performance(logger, 'info', 'Resource usage after preprocessing',
                    reset=True)

    return ops


def compute_drift_correction(ops, device, tic0=np.nan, progress_bar=None,
                             file_object=None, clear_cache=False, verbose=False):
    """Compute drift correction parameters and save them to `ops`.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.
    progress_bar : TODO; optional.
        Informs `tqdm` package how to report progress, type unclear.
    file_object : array-like file object; optional.
        Must have 'shape' and 'dtype' attributes and support array-like
        indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
        array or memmap.
    clear_cache : bool; False.
        If True, force pytorch to clear cached cuda memory after some
        memory-intensive steps in the pipeline.
    verbose : bool; False.
        If true, include additional debug-level logging statements.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.
    st0 : np.ndarray.
        Intermediate spike times variable with 6 columns. This is only used
        for generating the 'Drift Scatter' plot through the GUI.
    
    """

    tic = time.time()
    logger.info(' ')
    logger.info('Computing drift correction.')
    logger.info('-'*40)

    n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
        _, _, tmin, tmax, artifact, shift, scale, batch_downsampling = \
            get_run_parameters(ops)
    hp_filter = ops['preprocessing']['hp_filter']
    whiten_mat = ops['preprocessing']['whiten_mat']
    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device, do_CAR=do_CAR,
        invert_sign=invert, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, shift=shift, scale=scale,
        file_object=file_object, batch_downsampling=batch_downsampling
        )

    ops, st = datashift.run(ops, bfile, device=device, progress_bar=progress_bar,
                            clear_cache=clear_cache, verbose=verbose)
    
    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_drift'] = elapsed
    ops['usage_drift'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_drift'] = torch.cuda.memory_stats()
    logger.info(f'drift computed in {elapsed:.2f}s; total {total:.2f}s')

    if st is not None:
        logger.debug(f'st shape: {st.shape}')
        logger.debug(f'yblk shape: {ops["yblk"].shape}')
        logger.debug(f'dshift shape: {ops["dshift"].shape}')
        logger.debug(f'iKxx shape: {ops["iKxx"].shape}')
    
    # binary file with drift correction
    bfile = io.BinaryFiltered(
        ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
        hp_filter=hp_filter, whiten_mat=whiten_mat, device=device,
        dshift=ops['dshift'], do_CAR=do_CAR, dtype=dtype, tmin=tmin, tmax=tmax,
        artifact_threshold=artifact, shift=shift, scale=scale,
        file_object=file_object, batch_downsampling=batch_downsampling
        )

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after drift correction',
                    reset=True)

    return ops, bfile, st


def detect_spikes(ops, device, bfile, tic0=np.nan, progress_bar=None,
                  clear_cache=False, verbose=False):
    """Detect spikes via template deconvolution.
    
    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.
    progress_bar : TODO; optional.
        Informs `tqdm` package how to report progress, type unclear.
    clear_cache : bool; False.
        If True, force pytorch to clear cached cuda memory after some
        memory-intensive steps in the pipeline.
    verbose : bool; False.
        If true, include additional debug-level logging statements.

    Returns
    -------
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).

    """

    tic = time.time()
    logger.info(' ')
    logger.info(f'Extracting spikes using templates')
    logger.info('-'*40)
    st0, tF, ops = spikedetect.run(
        ops, bfile, device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    tF = torch.from_numpy(tF)

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_st0'] = elapsed
    ops['usage_st0'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_st0'] = torch.cuda.memory_stats(device)
    logger.info(f'{len(st0)} spikes extracted in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'st0 shape: {st0.shape}')
    logger.debug(f'tF shape: {tF.shape}')
    if len(st0) == 0:
        raise ValueError('No spikes detected, cannot continue sorting.')
    log_performance(logger, 'info', 'Resource usage after spike detect (univ)',
                    reset=True)
    log_thread_count(logger)

    tic = time.time()
    logger.info(' ')
    logger.info('First clustering')
    logger.info('-'*40)
    clu, Wall = clustering_qr.run(
        ops, st0, tF, mode='spikes', device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    Wall3 = template_matching.postprocess_templates(
        Wall, ops, clu, st0, tF, device=device
        )

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_clu0'] = elapsed
    ops['usage_clu0'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_clu0'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')
    log_performance(logger, 'info', 'Resource usage after first clustering',
                    reset=True)
   
    log_thread_count(logger)

    tic = time.time()
    logger.info(' ')
    logger.info('Extracting spikes using cluster waveforms')
    logger.info('-'*40)
    st, tF, ops = template_matching.extract(
        ops, bfile, Wall3, device=device, progress_bar=progress_bar
        )
   
    log_thread_count(logger)

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_st'] = elapsed
    ops['usage_st'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_st'] = torch.cuda.memory_stats(device)
    logger.info(f'{len(st)} spikes extracted in {elapsed:.2f}s; ' +
                f'total {total:.2f}s')
    logger.debug(f'st shape: {st.shape}')
    logger.debug(f'tF shape: {tF.shape}')
    logger.debug(f'iCC shape: {ops["iCC"].shape}')
    logger.debug(f'iU shape: {ops["iU"].shape}')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after spike detect (learned)',
                    reset=True)

    return st, tF, Wall, clu


def cluster_spikes(st, tF, ops, device, bfile, tic0=np.nan, progress_bar=None,
                   clear_cache=False, verbose=False):
    """Cluster spikes using graph-based methods.
    
    Parameters
    ----------
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device
        Indicates whether `pytorch` operations should be run on cpu or gpu.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.
    progress_bar : TODO; optional.
        Informs `tqdm` package how to report progress, type unclear.
    clear_cache : bool; False.
        If True, force pytorch to clear cached cuda memory after some
        memory-intensive steps in the pipeline.
    verbose : bool; False.
        If True, include additional debug-level logging statements.

    Returns
    -------
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    
    """
    tic = time.time()
    logger.info(' ')
    logger.info('Final clustering')
    logger.info('-'*40)
    clu, Wall = clustering_qr.run(
        ops, st, tF,  mode = 'template', device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )
    
    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_clu'] = elapsed
    ops['usage_clu'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_clu'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')
    log_thread_count(logger)

    tic = time.time()
    logger.info(' ')
    logger.info('Merging clusters')
    logger.info('-'*40)
    Wall, clu, is_ref, st, tF = template_matching.merging_function(
        ops, Wall, clu, st, tF, device=device, check_dt=True
        )
    clu = clu.astype('int32')

    elapsed = time.time() - tic
    total = time.time() - tic0
    ops['runtime_merge'] = elapsed
    ops['usage_merge'] = get_performance()
    if torch.cuda.is_available():
        ops['cuda_merge'] = torch.cuda.memory_stats(device)
    logger.info(f'{clu.max()+1} units found, in {elapsed:.2f}s; ' + 
                f'total {total:.2f}s')
    logger.debug(f'clu shape: {clu.shape}')
    logger.debug(f'Wall shape: {Wall.shape}')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after clustering',
                    reset=True)

    return clu, Wall, st, tF


def save_sorting(ops, results_dir, st, clu, tF, Wall, imin, tic0=np.nan,
                 save_extra_vars=False, save_preprocessed_copy=False,
                 skip_dat_path=False):  
    """Save sorting results, and format them for use with Phy

    Parameters
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    results_dir : pathlib.Path
        Directory where results should be saved.
    st : np.ndarray
        3-column array of peak time (in samples), template, and thresold
        amplitude for each spike.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st[:,0]`.
    tF : torch.Tensor
        PC features for each spike, with shape
        (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    imin : int
        Minimum sample index used by BinaryRWFile, exported spike times will
        be shifted forward by this number.
    tic0 : float; default=np.nan.
        Start time of `run_kilosort`.
    save_extra_vars : bool; default=False.
        If True, save tF and Wall to disk along with copies of st, clu and
        amplitudes with no postprocessing applied.
    save_preprocessed_copy : bool; default=False.
        If True, save a pre-processed copy of the data (including drift
        correction) to `temp_wh.dat` in the results directory and format Phy
        output to use that copy of the data.
    skip_dat_path : bool; default=False.
        If True, will save `dat_path = 'no_path.bin'` in `params.py` in place
        of a real filename. This is done to prevent an error in Phy when filename
        has an unexpected format, like when using a `file_object` loaded from
        an external data format through SpikeInterface. The full filename(s) will
        still be included in `params.py` for reference, but will be commented out.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    similar_templates : np.ndarray.
        Similarity score between each pair of clusters, computed as correlation
        between clusters. Shape (n_clusters, n_clusters).
    is_ref : np.ndarray.
        1D boolean array with shape (n_clusters,) indicating whether each
        cluster is refractory.
    est_contam_rate : np.ndarray.
        Contamination rate for each cluster, computed as fraction of refractory
        period violations relative to expectation based on a Poisson process.
        Shape (n_clusters,).
    kept_spikes : np.ndarray.
        Boolean mask with shape (n_spikes,) that is False for spikes that were
        removed by `kilosort.postprocessing.remove_duplicate_spikes`
        and True otherwise.

    Notes
    -----
    For documentation of saved files, see `kilosort.io.save_to_phy`.

    """

    tic = time.time()
    logger.info(' ')
    logger.info('Saving to phy and computing refractory periods')
    logger.info('-'*40)
    results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes = \
        io.save_to_phy(
            st, clu, tF, Wall, ops['probe'], ops, imin, results_dir=results_dir,
            data_dtype=ops['data_dtype'], save_extra_vars=save_extra_vars,
            save_preprocessed_copy=save_preprocessed_copy,
            skip_dat_path=skip_dat_path
            )
    logger.info(f'{int(is_ref.sum())} units found with good refractory periods')
    
    ops['n_units_total'] = np.unique(clu).size
    ops['n_units_good'] = int(is_ref.sum())
    ops['n_spikes'] = st[kept_spikes].shape[0]
    if ops.get('dshift', None) is not None:
        ops['mean_drift'] = np.abs(ops['dshift']).mean(axis=0)[0]
    else:
        ops['mean_drift'] = np.nan

    elapsed = elapsed = time.time() - tic
    ops['runtime_postproc'] = elapsed
    ops['usage_postproc'] = get_performance()
    logger.info(f'Exporting to Phy took: {elapsed:.2f}s')

    runtime = time.time()-tic0
    seconds = runtime % 60
    mins = runtime // 60
    hrs = mins // 60
    mins = mins % 60

    logger.info(f'Total runtime: {runtime:.2f}s = {int(hrs):02d}:' +
                f'{int(mins):02d}:{round(seconds)} h:m:s')
    ops['runtime'] = runtime 
    io.save_ops(ops, results_dir)
    logger.info(f'Sorting output saved in: {results_dir}.')

    log_cuda_details(logger)
    log_performance(logger, 'info', 'Resource usage after saving',
                    reset=True)

    return ops, similar_templates, is_ref, est_contam_rate, kept_spikes


def load_sorting(results_dir, device=None, load_extra_vars=False):
    '''Load saved sorting results into memory.
    
    Parameters
    ----------
    results_dir : str or pathlib.Path
        Directory where results were saved.
    device : torch.device; optional.
        CPU or GPU device to use to load Pytorch tensors. By default, PyTorch
        will use the first detected GPU. If no GPUs are detected, CPU will be
        used. To set this manually, specify `device = torch.device(<device_name>)`.
        See PyTorch documentation for full description.
    load_extra_vars : default=False.
        If True, load tF, Wall, and full copies of st, clu, and spike amplitudes
        in addition to the other variables.

    Returns
    -------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        1D vector of spike times (in samples) for all clusters. This is *only*
        the first column of the 3-column array returned by `run_kilosort`.
    clu : np.ndarray
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st`.
    similar_templates : np.ndarray.
        Similarity score between each pair of clusters, computed as correlation
        between clusters. Shape (n_clusters, n_clusters).
    is_ref : np.ndarray.
        1D boolean array with shape (n_clusters,) indicating whether each
        cluster is refractory.
    est_contam_rate : np.ndarray.
        Contamination rate for each cluster, computed as fraction of refractory
        period violations relative to expectation based on a Poisson process.
        Shape (n_clusters,).
    kept_spikes : np.ndarray.
        Boolean mask with shape (n_spikes,) that is False for spikes that were
        removed by `kilosort.postprocessing.remove_duplicate_spikes`
        and True otherwise.
    tF : torch.Tensor.
        Only returned if `load_extra_vars` is True.
        PC features for each spike, with shape (n_spikes, nearest_chans, n_pcs)
    Wall : torch.Tensor.
        Only returned if `load_extra_vars` is True.
        PC feature representation of spike waveforms for each cluster, with shape
        (n_clusters, n_channels, n_pcs).
    full_st : np.ndarray.
        Only returned if `load_extra_vars` is True.
        3-column array of peak time (in samples), template, and threshold amplitude for
        each spike.
        Includes spikes removed by `kilosort.postprocessing.remove_duplicate_spikes`.
    full_clu : np.ndarray.
        Only returned if `load_extra_vars` is True.
        1D vector of cluster ids indicating which spike came from which cluster,
        same shape as `st[:,0]`.
        Includes spikes removed by `kilosort.postprocessing.remove_duplicate_spikes`.
    full_amp : np.ndarray.
        Only returned if `load_extra_vars` is True.
        Per-spike amplitudes, computed as the L2 norm of the PC features
        for each spike.
        Includes spikes removed by `kilosort.postprocessing.remove_duplicate_spikes`.
    
    '''
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    results_dir = Path(results_dir)
    ops = io.load_ops(results_dir / 'ops.npy', device=device)
    similar_templates = np.load(results_dir / 'similar_templates.npy')

    clu = np.load(results_dir / 'spike_clusters.npy')
    st = np.load(results_dir / 'spike_times.npy')
    kept_spikes = np.load(results_dir / 'kept_spikes.npy')
    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    is_ref, est_contam_rate = CCG.refract(clu, st / ops['fs'],
                                          acg_threshold=acg_threshold,
                                          ccg_threshold=ccg_threshold)

    results = [ops, st, clu, similar_templates, is_ref,
               est_contam_rate, kept_spikes]

    if load_extra_vars:
        # NOTE: tF and Wall always go on CPU, not CUDA
        tF = np.load(results_dir / 'tF.npy')
        tF = torch.from_numpy(tF)
        Wall = np.load(results_dir / 'Wall.npy')
        Wall = torch.from_numpy(Wall)
        full_st = np.load(results_dir / 'full_st.npy')
        full_clu = np.load(results_dir / 'full_clu.npy')
        full_amp = np.load(results_dir / 'full_amp.npy')
        results.extend([tF, Wall, full_st, full_clu, full_amp])

    return results
