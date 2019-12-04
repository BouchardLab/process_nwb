import argparse, glob, os, h5py
import numpy as np

from pynwb import NWBHDF5IO
from process_nwb.resample import resample
from process_nwb.tokenize import (load_bad_electrodes, load_anatomy, load_bad_times,
                                  get_speak_event, extract_windows)



parser = argparse.ArgumentParser(description='Tokenize and concatenate nwbs in a folder.')
parser.add_argument('folder', type=str, help='Folder')
parser.add_argument('--frequency', type=float, default=200., help='Frequency to resample to.')
args = parser.parse_args()

folder = args.folder
frequency = args.frequency

files = sorted(glob.glob(os.path.join(folder, '*.nwb')))
with NWBHDF5IO(files[0], 'r') as io:
    nwb = io.read()
    bad_elects = load_bad_electrodes(nwb)
    electrode_labels = load_anatomy(nwb)
good_elects = np.ones(bad_elects.shape, dtype=bool)

for fname in sorted(files):
    with NWBHDF5IO(fname, 'a') as io:
        nwb = io.read()
    good_elects *= ~load_bad_electrodes(nwb)

precentral = np.array([ai == 'precentral' for ai in electrode_labels])
postcentral = np.array([ai == 'postcentral' for ai in electrode_labels])
vsmc_orig = np.logical_or(precentral, postcentral)

Xs = []
ys = []
blocks = []

for fname in sorted(files):
    block = int(os.path.split(fname)[1].split('B')[1].split('.')[0])
    with NWBHDF5IO(fname, 'a') as io:
        nwb = io.read()

        electrode_labels = load_anatomy(nwb)
        bad_times = load_bad_times(nwb)
        precentral = np.array([ai == 'precentral' for ai in electrode_labels])
        postcentral = np.array([ai == 'postcentral' for ai in electrode_labels])
        vsmc = np.logical_or(precentral, postcentral)

        if not np.array_equal(vsmc, vsmc_orig):
            raise ValueError('vSMC mismatch across blocks', fname)

        pre = nwb.processing['preprocessing']
        wvlt = pre.data_interfaces['wvlt_amp_CAR_ln_downsampled_ElectricalSeries'].data[:]
        old_rate = pre.data_interfaces['wvlt_amp_CAR_ln_downsampled_ElectricalSeries'].rate
        wvlt = wvlt[:, np.logical_and(good_elects, vsmc)]
        wvlt = resample(wvlt, frequency, old_rate)

        event_times, event_labels = get_speak_event(nwb, 1)
        event_labels = np.array(event_labels, dtype='S4')
        block = block * np.ones(event_labels.size, dtype=int)
        X = extract_windows(wvlt, frequency, event_times, align_window=np.array([-0.5, 0.79]),
                            bad_times=bad_times)
        Xs.append(X)
        ys.append(event_labels)
        blocks.append(block)

Xs = np.concatenate(Xs, axis=0)
ys = np.concatenate(ys, axis=0)
blocks = np.concatenate(blocks, axis=0)


name = os.path.split(files[0])[1].split('_')[0]
with h5py.File(os.path.join(folder, '{}.h5'.format(name)), 'w') as f:
    f.create_dataset('X', data=Xs.astype('float32'))
    f.create_dataset('y', data=ys)
    f.create_dataset('block', data=blocks)
