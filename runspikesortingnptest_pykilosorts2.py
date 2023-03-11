# Set logging before the rest as neo (and neo-based imports) needs to be imported after logging has been set
import logging
import os

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('sorting')
logger.setLevel(logging.DEBUG)

from pathlib import Path
import datetime
import argparse
import json
import jsmin
from jsmin import jsmin
import numpy as np
os.environ['NUMEXPR_MAX_THREADS'] = '36'

import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.curation as scu
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp

import spikesorting_scripts
from spikesorting_scripts.postprocessing import postprocessing_si
from spikesorting_scripts.npyx_metadata_fct import get_npix_sync

def pad_amplitude(spike_time, amplitudes):
    padded_amplitudes = np.zeros_like(spike_time)
    padded_amplitudes[:len(amplitudes)] = amplitudes
    return padded_amplitudes


def spikeglx_preprocessing(recording):
    # Preprocessing steps
    logger.info(f'preprocessing recording')

    # equivalent to what catgt does
    recording = spre.phase_shift(recording)
    # bandpass filter and common reference can be skipped if using kilosort as it does it internally
    # but doesn't change anything to keep it
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = spre.common_reference(recording, reference='global', operator='median')
    return recording


def spikesorting_pipeline(rec_name, params):
    # Spikesorting pipeline for a single recording
    working_directory = Path(params['working_directory']) / 'tempDir'

    recording = se.read_spikeglx(rec_name, stream_id='imec0.ap')
    recording = spikeglx_preprocessing(recording)

    logger.info(f'running spike sorting')
    sorting_output = ss.run_sorters(params['sorter_list'], [recording], working_folder=working_directory,
                                    mode_if_folder_exists='keep',
                                    engine='loop', sorter_params= {'pykilosort': {'n_jobs': 18, 'chunk_size': 30000}}, verbose=True)


def spikesorting_postprocessing(params):
    jobs_kwargs = params['jobs_kwargs']
    # sorting_output = ss.collect_sorting_outputs(Path(params['working_directory']))
    # for (rec_name, sorter_name), sorting in sorting_output.items():
    #     logger.info(f'Postprocessing {rec_name} {sorter_name}')
    #     if params['remove_dup_spikes']:
    #         logger.info(f'removing duplicate spikes')
    #         sorting = scu.remove_duplicated_spikes(sorting, censored_period_ms=params['remove_dup_spikes_params']['censored_period_ms'])

    #     logger.info('waveform extraction')
    #     outDir = Path(params['output_folder']) / rec_name / sorter_name
    #     we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder',
    #             overwrite=True,
    #             ms_before=1, ms_after=2., max_spikes_per_unit=100,
    #             **jobs_kwargs)

    #     logger.info(f'Computing quality metrics')
    #     metrics = sqm.compute_quality_metrics(we, n_jobs = jobs_kwargs['n_jobs'], verbose=True)

    #     logger.info(f'Exporting to phy')
    #     sexp.export_to_phy(we, outDir / 'phy_folder',
    #         verbose=True, 
    #         compute_pc_features=False,
    #         **jobs_kwargs)
        
    #     postprocessing_si(outDir / 'phy_folder')

    #     spike_times = np.load(outDir / 'phy_folder' / 'spike_times.npy')
    #     amplitudes = np.load(outDir / 'phy_folder' / 'amplitudes.npy')
    #     padded_amplitudes_array = pad_amplitude(spike_times, amplitudes)
    #     np.save('/home/zceccgr/Scratch/zceccgr/neuropixelsdecodingproject/results_pykilo_s2/recording_0/pykilosort/sorter_output/output/amplitudes.npy', padded_amplitudes_array)
    #     np.save('/home/zceccgr/Scratch/zceccgr/neuropixelsdecodingproject/results_pykilo_s2/recording_0/pykilosort/waveforms_folder/spike_amplitudes/amplitude_segment_0.npy', padded_amplitudes_array)
    sorting_output = ss.collect_sorting_outputs(Path(params['working_directory']))
    for (rec_name, sorter_name), sorting in sorting_output.items():
        logger.info(f'Postprocessing {rec_name} {sorter_name}')
        # if params['remove_dup_spikes']:
        #     logger.info(f'removing duplicate spikes')
        #     sorting = scu.remove_duplicated_spikes(sorting, censored_period_ms=params['remove_dup_spikes_params']['censored_period_ms'])

        logger.info('waveform extraction')
        outDir = Path(params['output_folder']) / rec_name / sorter_name
        we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder', load_if_exists=True,
                ms_before=1, ms_after=2., max_spikes_per_unit=100,
                **jobs_kwargs)

        logger.info(f'Computing quality metrics')
        # with PCs
        from spikeinterface.postprocessing import compute_principal_components
        # pca = compute_principal_components(we, n_components=5, mode='by_channel_local')
        print('computing quality metrics')
        metrics = sqm.compute_quality_metrics(we,  metric_names =[
        "num_spikes",
        "firing_rate",
        "presence_ratio",
        "snr",
        "isi_violation",
        "rp_violation",
        "sliding_rp_violation",
        "amplitude_cutoff",
        "amplitude_median",
        "isolation_distance",
        "l_ratio",
        "d_prime",
        "nearest_neighbor"
    ],)
        assert 'isolation_distance' in metrics.columns
        # logger.info(f'Exporting to phy')
        # sexp.export_to_phy(we, outDir / 'phy_folder2',
        #     verbose=True, 
        #     compute_pc_features=False,
        #     **jobs_kwargs)

        # print(we)

        logger.info('Export report')
        print('exporting report')
        sexp.export_report(we, outDir / 'report2', format='png', force_computation=False, **jobs_kwargs)

        # try:
        #     logger.info('Export report')
        #     sexp.export_report(we, outDir / 'report', padded_amplitudes_array, format='png', force_computation=False, use_padded_amplitudes=False, **jobs_kwargs)
        # except Exception as e:
        #     logger.warning(f'Export report failed: {e}')


def main():
    # parser = argparse.ArgumentParser()
    params_file = Path(__file__).parent / 'my_silly_params_pykilosorts2.json'
    # parser.add_argument("params_file", help="path to the json file containing the parameters")
    # args.params_file = params_file
    # args = parser.parse_args()

    with open(params_file) as json_file:
        minified = jsmin(json_file.read())  # Parses out comments.
        params = json.loads(minified)

    logpath = Path(params['logpath'])
    now = datetime.datetime.now().strftime('%d-%m-%Y_%H_%M_%S')

    fh = logging.FileHandler(logpath / f'neuropixels_sorting_logs_{now}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Starting')

    sorter_list = params['sorter_list']  # ['klusta'] #'kilosort2']
    logger.info(f'sorter list: {sorter_list}')

    if 'kilosort2' in sorter_list:
        ss.Kilosort2Sorter.set_kilosort2_path(params['sorter_paths']['kilosort2_path'])
    if 'waveclus' in sorter_list:
        ss.WaveClusSorter.set_waveclus_path(params['sorter_paths']['waveclus_path'])
    if 'kilosort3' in sorter_list:
        ss.Kilosort3Sorter.set_kilosort3_path(params['sorter_paths']['kilosort3_path'])

    datadir = Path(params['datadir'])
    output_folder = Path(params['output_folder'])
    working_directory = Path(params['working_directory'])

    logger.info('Start loading recordings')

    # Load recordings
    sessions = [sess.name for sess in datadir.glob('*_g0')]

    recordings_list = []
    # /!\ This assumes that all the recordings must have same mapping
    for session in sessions:
        # Extract sync onsets and save as catgt would
        get_npix_sync(datadir / session, sync_trial_chan=[5])

        recording = se.read_spikeglx(datadir / session, stream_id='imec0.ap')
        recording = spikeglx_preprocessing(recording)
        recordings_list.append(recording)
    
    multirecordings = sc.concatenate_recordings(recordings_list)
    multirecordings = multirecordings.set_probe(recordings_list[0].get_probe())
    logger.info('sorting now')
    sorting = ss.run_sorters(params['sorter_list'], [multirecordings], working_folder=working_directory,
                             mode_if_folder_exists='keep',
                             engine='loop', verbose=True)

    # # If recordings don't have same mapping, can do something like this:
    # # In this example, only 2 mappings are in the data, but it can be extended to more mappings
    # # To extract channel coordinates from a recording object, use recording.get_channel_locations()
    # # To extract channel coordinates from a probe object, use probe.get_channel_locations()
    # # And then group recordings based on this
    # # More information about probe object on https://probeinterface.readthedocs.io/en/main/

    # recordings_list_probemap_12 = []
    # recordings_list_probemap_34 = []

    # for session in sessions:
    #     # Extract sync onsets and save as catgt would
    #     get_npix_sync(datadir / session, sync_trial_chan=[5])

    #     recording = se.read_spikeglx(catgt_data / session, stream_id='imec0.ap')
    #     recording = spikeglx_preprocessing(recording)

    #     probe = recording.get_probe()
    #     if '0' in probe.shank_ids:
    #         recordings_list_probemap_12.append(recording)
    #     else:
    #         recordings_list_probemap_34.append(recording)

    # for (multirec, probemap_name) in zip(
    #     [recordings_list_probemap_12, recordings_list_probemap_34],['probemap_12', 'probemap_34']):
    # multirecordings = si.concatenate_recordings(multirec)
    # multirecordings = multirecordings.set_probe(multirec[0].get_probe())
    # multirecordings.is_filtered = True

    # sorting = si.run_kilosort3(multirecordings, output_folder=catgt_data / f'{probemap_name}_concatenated')

    # Not sure if it works with concatenated recordings
    # And might take a while to run extract waveforms
    spikesorting_postprocessing(params)


if __name__ == '__main__':
    main()