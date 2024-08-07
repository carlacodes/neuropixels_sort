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
import json
import jsmin
from jsmin import jsmin
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.curation as scu
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp
import spikeinterface.full as si
from helpers import pipeline_helpers
from spikeinterface.postprocessing import compute_principal_components
import torch
from spikeinterface.qualitymetrics import (compute_snrs, compute_firing_rates,
                                                    compute_isi_violations, calculate_pc_metrics,
                                                    compute_quality_metrics)

#set limit of 16 threads
from threadpoolctl import threadpool_limits




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
                                    engine='loop', sorter_params={'pykilosort': {'n_jobs': 19, 'chunk_size': 30000}},
                                    verbose=True)


def spikesorting_postprocessing(params, step_one_complete=False):
    jobs_kwargs = params['jobs_kwargs']
    if step_one_complete == False:
        sorting_output = ss.collect_sorting_outputs(Path(params['working_directory']))
        for (rec_name, sorter_name), sorting in sorting_output.items():
            logger.info(f'Postprocessing {rec_name} {sorter_name}')
            if params['remove_dup_spikes']:
                print('remove dup spikes')
                logger.info(f'removing duplicate spikes')
                sorting = scu.remove_duplicated_spikes(sorting, censored_period_ms=params['remove_dup_spikes_params'][
                    'censored_period_ms'])
                sorting = scu.remove_excess_spikes(sorting, sorting._recording)

            logger.info('waveform extraction')
            outDir = Path(params['output_folder']) / rec_name / sorter_name
            # we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder2', ms_before=1, ms_after=2., max_spikes_per_unit=300, n_jobs = -1, chunk_size=3000)
            we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder_sparse3',
                    load_if_exists=True,
                    # overwrite=False,
                    ms_before=2, 
                    ms_after=3., 
                    max_spikes_per_unit=300,
                    sparse=True,
                    num_spikes_for_sparsity=100,
                    method="radius",
                    radius_um=100,
                    **jobs_kwargs)            # we = sc.load_waveforms(outDir / 'waveforms_sparse_folder')
            logger.info(f'Computing quality metrics')
            # with PCs

            pca = compute_principal_components(we, n_components=3, mode='by_channel_local')
            # logger.info('compute lratio')
            #changed number of jobs to 1 as was runing out of space from parallel computation
            metrics = compute_quality_metrics(we, metric_names=['d_prime'], n_jobs=8)
            # metrics = compute_quality_metrics(we)
            logger.info('Export report')
            print('exporting report')
            sexp.export_report(we, outDir / 'report2', format='png', force_computation=False, **jobs_kwargs)
            #n_jobs = 8, chunk_size=3000
            # logger.info(f'Exporting to phy')
            # sexp.export_to_phy(we, outDir / 'phy5_folder', remove_if_exists=True,
            #                 verbose=True,
            #                 compute_pc_features=False,
            #                 **jobs_kwargs) 



    

        # try:
        #     logger.info('Export report')
        #     sexp.export_report(we, outDir / 'report', padded_amplitudes_array, format='png', force_computation=False, use_padded_amplitudes=False, **jobs_kwargs)
        # except Exception as e:
        #     logger.warning(f'Export report failed: {e}')


def main():
    # parser = argparse.ArgumentParser()
    ##checking if torch device is available
    with threadpool_limits(limits=16, user_api='blas'):
        logger.info('loading torch')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(device)


        params_file = Path('/nfs/nhome/live/carlag/neuropixels_sort/params/rat_params_hc2_16072024.json')  # 'params/params.json
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
        day_folder = datadir.name
        rat_folder = datadir.parent.name
        output_folder = Path(params['output_folder'])
        #concatenate the rat_folder and day_folder to the output_folder
        output_folder = str(output_folder)+ f'_{rat_folder}'+ f'_{day_folder}'
        output_folder = Path(output_folder)

        #check if output_folder exits
        if output_folder.exists() == False:
        # working_directory = Path(params['working_directory'])

            logger.info('Start loading recordings')
            sessions = [f.name for f in datadir.iterdir() if f.is_dir()]
            print('sessions are:')
            print(sessions)

            recordings_list = []
            # /!\ This assumes that all the recordings must have same mapping, pretty sure I was reading using the IBL cbin function after compressing the data
            for session in sessions:
                # Extract sync onsets and save as catgt would
                # get_npix_sync(datadir / session, sync_trial_chan=[5])
                logger.info(session)
                print(session)
                imec0_file = session + '_imec0'

                try:
                    recording = se.read_spikeglx(datadir / session, stream_id='imec0.ap')
                    print(datadir / session/ imec0_file)
                except Exception as e:
                    print(f'error loading recording: {e}, going to ibl file')
                    recording = se.read_cbin_ibl(datadir / session/ imec0_file)
                recording = spikeglx_preprocessing(recording)
                recordings_list.append(recording)
                # except:
                #     print('issue preprocessing:'+ session)

            multirecordings = sc.concatenate_recordings(recordings_list)
            multirecordings = multirecordings.set_probe(recordings_list[0].get_probe())
            #save the multirecordings
            # logger.info('saving multirecordings')
            # job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)

            # multirecordings.save(folder = output_folder, **job_kwargs)
            logger.info('sorting now')
            sorting = ss.run_sorter(sorter_name="kilosort4", recording=multirecordings, output_folder=output_folder, batch_size = 60000, verbose=True)
            pipeline_helpers.spikesorting_postprocessing(sorting, output_folder, datadir)
            logger.info('Postprocessing done')
        else:
            logger.info('Output folder already exists, skipping sorting and trying postprocessing')

            sorting = si.read_sorter_folder(output_folder)
            pipeline_helpers.spikesorting_postprocessing(sorting, output_folder, datadir)
            logger.info('Postprocessing done')

if __name__ == '__main__':
    main()