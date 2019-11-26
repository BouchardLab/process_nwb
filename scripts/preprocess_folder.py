import argparse, glob, os

from pynwb import NWBHDF5IO
from process_nwb.resample import store_resample

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
        store_resample(nwbfile, 'ElectricalSeries', frequency)

        print(nwbfile.acquisition)
