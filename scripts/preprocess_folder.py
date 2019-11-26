import argparse, glob, os

from pynwb import NWBHDF5IO

from process_nwb.resample import store_resample
from process_nwb import store_linenoise_notch_CAR


parser = argparse.ArgumentParser(description='Preprocess nwbs in a folder.')
parser.add_argument('folder', type=str, help='Folder')
parser.add_argument('--frequency', type=float, default=400., help='Frequency to resample to.')
args = parser.parse_args()

folder = args.folder
frequency = args.frequency

files = glob.glob(os.path.join(folder, '*.nwb'))
for fname in files:
    with NWBHDF5IO(fname, 'r') as io:
        nwbfile = io.read()
        electrical_series = nwbfile.acquisition['ElectricalSeries']
        nwbfile.create_processing_module(name='preprocessing',
                                         description='Preprocessing.')
        _, electrical_series_ds = store_resample(electrical_series,
                                                 nwbfile.processing['preprocessing'],
                                                 frequency)
        _, electrical_series_CAR = store_linenoise_notch_CAR(electrical_series_ds,
                                                      nwbfile.processing['preprocessing'])

        print(nwbfile.acquisition)
        print()
        print(nwbfile.processing)
        print()
        print()
        print()
