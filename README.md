# process_nwb

[![Actions Status](https://github.com/BouchardLab/process_nwb/workflows/process_nwb%20tests/badge.svg)](https://github.com/BouchardLab/process_nwb/actions) [![codecov](https://codecov.io/gh/BouchardLab/process_nwb/branch/master/graph/badge.svg)](https://codecov.io/gh/BouchardLab/process_nwb)



Functions for preprocessing (ECoG) data stored in the NWB format


## Installation

process_nwb is availble on PyPI

```bash
pip install process-nwb
```

### From source
To install, you can clone the repository and `cd` into the process_nwb folder.

```bash
# use ssh
$ git clone git@github.com:BouchardLab/process_nwb.git
# or use https
$ git clone https://github.com/BouchardLab/process_nwb.git
$ cd process_nwb
```

If you are installing into an active conda environment, you can run

```bash
$ conda env update --file environment.yml
$ pip install -e .
```

If you are installing with `pip` you can run

```bash
$ pip install -e . -r requirements.txt
```
