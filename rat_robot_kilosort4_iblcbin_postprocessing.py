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
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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



def main():
    # parser = argparse.ArgumentParser()
    ##checking if torch device is available
    logger.info('loading torch')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device:", device)


    params_file = Path('/nfs/nhome/live/carlag/neuropixels_sort/params/rat_params_12072024.json')  # 'params/params.json
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
    rat_dir = datadir.parent.name
    day_dir = datadir.name
    #concatenate the rat_folder and day_folder to the output_folder
    output_folder = Path(f'{str(output_folder)}_{rat_dir}_{day_dir}')
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

            # try:
            recording = se.read_spikeglx(datadir / session, stream_id='imec0.ap')
            print(datadir / session/ imec0_file)
            # recording = se.read_cbin_ibl(datadir / session/ imec0_file)
            recording = spikeglx_preprocessing(recording)
            recordings_list.append(recording)
            # except:
            #     print('issue preprocessing:'+ session)

        multirecordings = sc.concatenate_recordings(recordings_list)
        multirecordings = multirecordings.set_probe(recordings_list[0].get_probe())
        # #save the multirecordings
        # logger.info('saving multirecordings')
        # # job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)

        # multirecordings.save(folder = output_folder, **job_kwargs)
        logger.info('sorting now')
        sorting = ss.run_sorter(sorter_name="kilosort4", recording=multirecordings, output_folder=output_folder, batch_size = 60000, verbose=True)
    else:
        print('sorting path already exists')
        logger.info('Output folder already exists, skipping sorting and trying postprocessing')
        sorting = si.read_sorter_folder(output_folder)
        pipeline_helpers.spikesorting_postprocessing(sorting, output_folder, datadir)
        logger.info('Postprocessing done')

if __name__ == '__main__':
    main()