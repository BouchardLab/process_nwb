.. process_nwb

======================================================
Introduction: Processing ECoG stored in the NWB format
======================================================

`process_nwb` is meant to be an experiment-agnostic library for preprocessing
ECoG data stored in the Neurodata Without Borders (NWB) format. It
contains low-level signal processing routines and wrapper functions to interface
with the NWB format.

These functions can be used individually but they are also packaged into an executable script
`preprocess_folder`. This can be used to preprocess all nwb files in a given folder. Preprocessing
consists of

  1. downsampling the original signal (to 3200 Hz by default) and storing this new signal in the file,
  2. notch filter for 60 Hz linenoise and calculate, store, and remove a common average reference (mean of 95% by default), the new signal is stored
  3. calculate and potentially resample the amplitudes for a filterbank of wavelets, by default constant-Q and log-spaced center frequencies. By default, only the amplitudes are stored.

Run

.. code-block:: console

    preprocess_folder -h

for options.
