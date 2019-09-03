## TIME SCALE MODIFICATION OF AUDIO USING NON-NEGATIVE MATRIX FACTORIZATION

This repository contains the code to reproduce the method presented in:

    Roma, G., Green, O. & Tremblay, P. A., Time scale modification of audio using non-negative matrix factorization. Proceedings of the 22nd International Conference on Digital Audio Effects (DAFX 2019)


### Requirements:
- Python 3
- numpy, scipy
- untwist (https://github.com/IoSR-Surrey/untwist)
- scnmf (https://github.com/mmxgn/smooth-convex-kl-nmf)

### Usage
The main script is nmf_tsm.py:
```
python nmf_tsm.py input_file_name.wav stretch_factor nmf_rank <t1> <t2> <t3>
```
The first three arguments are mandatory. If nmf_rank is 0, the rank will be automatically estimated via singular value decomposition. You can also modify the defaults in the first lines in the script. The envelope preservation option can also be switched in the code (lock_active). For more information please see the paper.

### Other methods
Some classic time scale modification methods are included for comparison in classic_tsm.py (ported from the matlab TSM Toolbox).
Here's an example python code for using wsola with a 1.8 stretch factor, assuming the signal is read in the numpy arrray x:
y = wsola(x, 1.8, Hs = 512, window = signal.hann(2048, sym = False))
