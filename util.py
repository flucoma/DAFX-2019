import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import stft
from scipy import signal

"""
Utility functions
"""

def wrap_phase(phase):
    return phase - 2 * np.pi * np.round(phase / ( 2 *np.pi))

def get_phase_dev(Xp, phase_increment, frame):
    current_phase = Xp[:, frame]
    prev_phase = Xp[:, frame - 1]
    phase_dev = current_phase - prev_phase - phase_increment
    return wrap_phase(phase_dev)

def get_peak_phase_dev(Xp, phase_increment, frame, bin):
    current_phase = Xp[bin, frame]
    prev_phase = Xp[:, frame - 1]
    phase_dev = current_phase - prev_phase - phase_increment
    return wrap_phase(phase_dev)


def showspec(matrix):
    plt.figure()
    plt.imshow(np.flipud(matrix), aspect="auto")
    plt.colorbar()


def read_mono(path):
    with open(path, 'rb') as f:
        data, samplerate = sf.read(f)
    return data, samplerate

def write_mono(x, path, sr = 44100):
    sf.write(path, x, sr, 'PCM_24')

def render_estimate(est, sr, name="test.wav"):
    y = stft.istft(est, 256)
    write_mono(y, "result/"+name, sr)

def is_pitched(w, th):
    result =  np.correlate(w, w, mode='full')
    ac = result[int(result.size/2):]
    ac = ac - signal.medfilt(ac, 101)
    peaks = detect_peaks(ac[1:])
    max_peak = peaks[np.argmax(ac[peaks+1])]
    return np.max(ac[peaks+1]) > th

def detect_peaks(x, mph = None):
    dx = x[1:] - x[:-1]
    peaks = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    if peaks.size and peaks[0] == 0:
        peaks = peaks[1:]
    if peaks.size and peaks[-1] == x.size-1:
        peaks = peaks[:-1]
    if peaks.size and mph is not None:
        peaks = peaks[x[peaks] >= mph]
    return peaks

def get_peaks(Xm):
    peaks = detect_peaks(Xm)
    valleys = detect_peaks(-Xm)
    if(len(peaks)):
        if peaks[0] < valleys[0]:
            valleys = np.hstack((0, valleys))
        if peaks[-1] > valleys[-1]:
            valleys = np.hstack((valleys, len(Xm) - 1))
    return peaks, valleys


# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y
