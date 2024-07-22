from pathlib import Path
'''original author: Jules Lebert, modified by carla griffiths'''
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import logging
from threadpoolctl import threadpool_limits

logging.basicConfig(level=logging.DEBUG)

logging.debug("Before threadpool_limits")
# Limit the number of threads to 1 for libraries using OpenMP
with threadpool_limits(limits=12, user_api='blas'):
    # Your code that uses the conflicting libraries here
    pass
def spikeglx_preprocessing(recording):
    recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
    bad_channel_ids, channel_labels = si.detect_bad_channels(recording)
    recording = recording.remove_channels(bad_channel_ids)
    recording = si.phase_shift(recording)
    recording = si.common_reference(recording, reference='global', operator='median')

    return recording


def spikesorting_pipeline(recording, working_directory, sorter='kilosort4'):
    # working_directory = Path(output_folder) / 'tempDir' / recording.name

    if (working_directory / 'binary.json').exists():
        recording = si.load_extractor(working_directory)
    else:
        recording = spikeglx_preprocessing(recording)
        job_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)
        recording = recording.save(folder = working_directory, format='binary', **job_kwargs)

    sorting = si.run_sorter(
        sorter_name=sorter, 
        recording=recording, 
        output_folder = working_directory / f'{sorter}_output',
        verbose=True,
        )
    
    return sorting


def spikesorting_postprocessing(sorting, output_folder, datadir):
    output_folder.mkdir(exist_ok=True, parents=True)

    outDir = output_folder/ sorting.name

    jobs_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)
    print('removing dupliated spikes')
    sorting = si.remove_duplicated_spikes(sorting, censored_period_ms=2)
    rec = sorting._recording


    outDir = output_folder/ sorting.name

    jobs_kwargs = dict(n_jobs=18, chunk_duration='1s', progress_bar=True)
    sorting = si.remove_duplicated_spikes(sorting, censored_period_ms=2)

    if (outDir / 'waveforms_folder').exists():
        we = si.load_waveforms(
            outDir / 'waveforms_folder', 
            sorting=sorting,
            with_recording=True,
            )
        with threadpool_limits(limits=12, user_api='blas'):

            si.export_report(we, outDir / 'report',
                             format='png',
                             force_computation=True,
                             **jobs_kwargs,
                             )

    else:
        print('extractng waveforms')
        we = si.extract_waveforms(rec, sorting, outDir / 'waveforms_folder',
            overwrite=None,
            ms_before=2, 
            ms_after=3., 
            max_spikes_per_unit=500,
            sparse=True,
            num_spikes_for_sparsity=100,
            method="radius",
            radius_um=40,
            **jobs_kwargs,
            )

        logging.debug("Before threadpool_limits")

        with threadpool_limits(limits=12, user_api='blas'):
            logging.debug("Inside threadpool_limits")

            metric_list = si.get_quality_metric_list()
            metrics = si.compute_quality_metrics(
                we,
                n_jobs = jobs_kwargs['n_jobs'],
                verbose=True,
                )
            si.export_to_phy(we, outDir / 'phy_folder',
                            verbose=True,
                            compute_pc_features=False,
                            copy_binary=False,
                            remove_if_exists=False,
                            **jobs_kwargs,
                            )


            si.export_report(we, outDir / 'report',
                    format='png',
                    force_computation=True,
                    **jobs_kwargs,
                    )