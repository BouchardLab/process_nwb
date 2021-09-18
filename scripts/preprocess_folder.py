#! /usr/bin/env python
import argparse, glob, os
from hdmf.data_utils import InvalidDataIOError

from pynwb import NWBHDF5IO

from process_nwb.resample import store_resample
from process_nwb.wavelet_transform import store_wavelet_transform
from process_nwb import store_linenoise_notch_CAR


parser = argparse.ArgumentParser(description='Preprocess nwbs in a folder.' +
                                 '\nPerforms the following steps:' +
                                 '\n1) Resample to frequency and store result,' +
                                 '\n2) Remove 60Hz noise and remove and store the CAR, and' +
                                 '\n3) Perform and store a wavelet decomposition.')
parser.add_argument('folder', type=str, help='Folder')
parser.add_argument('--initial_resample_rate', type=float, default=3200., help='Frequency to resample '
                    'to before performing wavelet transform.')
parser.add_argument('--final_resample_rate', type=float, default=400., help='Frequency to resample '
                    'to after calculating wavelet amplitudes.')
parser.add_argument('--filters', type=str, default='rat',
                    choices=['rat', 'human', 'changlab'],
                    help='Type of filter bank to use for wavelets.')
parser.add_argument('--all_filters', action='store_true')
parser.add_argument('--acq_name', type=str, default='ECoG',
                    help='Name of acquisition in NWB file')
args = parser.parse_args()

folder = args.folder
initial_resample_rate = args.initial_resample_rate
final_resample_rate = args.final_resample_rate
filters = args.filters
hg_only = not args.all_filters
acq_name = str(args.acq_name)

files = glob.glob(os.path.join(folder, '*.nwb'))

if folder.endswith == '.nwb':
    raise Exception('Please specify the folder CONTAINING the NWB file, not the nwb file itself')

if len(files) == 0:
    raise Exception('No NWB files in folder or invalid folder path')

for fname in files:
    print('Processing {}'.format(fname))
    with NWBHDF5IO(fname, 'a') as io:
        nwbfile = io.read()
        electrical_series = nwbfile.acquisition[acq_name]
        nwbfile.create_processing_module(name='preprocessing',
                                         description='Preprocessing.')

        _, electrical_series_ds = store_resample(electrical_series,
                                                 nwbfile.processing['preprocessing'],
                                                 initial_resample_rate)
        del _

        _, electrical_series_CAR = store_linenoise_notch_CAR(electrical_series_ds,
                                                             nwbfile.processing['preprocessing'])
        del _

        _, electrical_series_wvlt = store_wavelet_transform(electrical_series_CAR,
                                                            nwbfile.processing['preprocessing'],
                                                            filters=filters,
                                                            hg_only=hg_only,
                                                            post_resample_rate=final_resample_rate)
        del _

        io.write(nwbfile)
