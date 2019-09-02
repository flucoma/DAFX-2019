
import sys
from os.path import splitext, basename
import numpy as np
from scipy import signal
from untwist.transforms import STFT, ISTFT
from untwist.factorizations import NMF
from untwist.data import Wave
from untwist.data import Spectrogram
from util import *
from segment_activations import *
from scnmf import *
from nmf import *

eps = np.spacing(1)
fft_size = 1024
hop_size = 256
window_size = 1024
rank = 4
t1 = 0.1
t2 = -0.2
t3 = 0.5
n_bins = fft_size/2 + 1

in_fname = str(sys.argv[1])
factor = float(sys.argv[2])
rank = float(sys.argv[3])
if len(sys.argv) > 4: t1 = float(sys.argv[4])
if len(sys.argv) > 5: t2 = float(sys.argv[5])
if len(sys.argv) > 6: t3 = float(sys.argv[6])

out_fname = splitext(in_fname)[0]+"_"+str(factor)+".wav"

stft = STFT(signal.hann(window_size, sym = False), fft_size,hop_size)
istft =  ISTFT(signal.hann(window_size, sym = False), fft_size,hop_size)
x = Wave.read(in_fname)
if len(x.shape) > 1 and x.shape[1] > 1:
    x = x[:,0]
sr = x.sample_rate
X = stft.process(x)

Xm = np.abs(X)
Xp = np.angle(X)

radian_bin_freqs = 2 * np.pi * np.arange(Xm.shape[0]) / fft_size
phase_increment = radian_bin_freqs * hop_size

phase_lock = True
lock_active = True # change for envelope preservation

if rank < 1:
    rank, W0, H0 = init_nmf(Xm)
else:
    rank = int(rank)

V = Xm + eps
print("computing NMF")
L, H, E = smoothConvexNMF(V, rank, beta = 0.01, max_iter = 300)
W = np.dot(V, L)
n_in_frames = Xm.shape[1]
n_out_frames = int(np.round(n_in_frames * factor))
track = 0
stretched = None
for track in range(rank):
    activation = H[track,:]
    an_events = make_analysis_events(activation, t1, t2, t3)
    syn_events = make_synthesis_events(an_events, factor, lock_active)
    if len(an_events) == 0 or len(syn_events) == 0:
        continue
    X1 = X * get_mask(track, W, H)
    X1m = np.abs(X1)
    Ym = np.zeros((Xm.shape[0], int(n_out_frames)))
    Yp = np.zeros((Xm.shape[0], int(n_out_frames)))

    for a, s in zip(an_events, syn_events):
        # copy transient
        Ym[:, s.start:s.start + s.n_transient] = X1m[:,a.start:a.start + s.n_transient] # using s.n_transient as it could be smaller
        Yp[:, s.start:s.start + s.n_transient] = Xp[:,a.start:a.start + s.n_transient]
        prev_syn_phase = Yp[:,s.start + s.n_transient - 1]
        # copy active
        if s.n_active > 0:
            act_rate = a.n_active / s.n_active
            an_ptr = a.start + a.n_transient
            for i in range(s.start + s.n_transient, s.start + s.n_transient + s.n_active):
                an_frame = np.round(an_ptr).astype(int)
                if an_frame >=  X1m.shape[1]: an_frame =  X1m.shape[1] - 1
                an_frame1 = min(np.floor(an_ptr).astype(int), X1m.shape[1] - 1)
                an_frame2 = min(np.ceil(an_ptr).astype(int), X1m.shape[1] - 1)
                frac = an_ptr - np.floor(an_ptr)
                Ym[:,i] = (1-frac) * X1m[:, an_frame1] + frac * X1m[:, an_frame2]
                phase_dev = get_phase_dev(Xp, phase_increment, an_frame)
                if phase_lock:
                    p, v = get_peaks(Ym[:,i])
                    for j in range(len(p)):
                            phase_env = wrap_phase(Xp[v[j]:v[j+1],an_frame] - Xp[p[j], an_frame] + phase_dev[p[j]])
                            Yp[v[j]:v[j+1], i]  = prev_syn_phase[p[j]] + phase_increment[p[j]] + phase_env
                else:
                    phase_dev = get_phase_dev(Xp, phase_increment, an_frame)
                    Yp[:,i] = prev_syn_phase + phase_increment + phase_dev
                    Yp[:,i] = Xp[:, an_frame]
                prev_syn_phase = Yp[:,i]
                an_ptr = an_ptr + act_rate
        # copy silence
        if s.n_silence > 0:
            rest_rate = a.n_silence / s.n_silence
            an_ptr = a.start + a.n_transient + a.n_active
            mag_rate = X1m[:,an_ptr] - X1m[:,an_ptr-1]
            mag_incr = mag_rate
            for i in range(s.start + s.n_transient + s.n_active, s.end):
                    an_frame = np.round(an_ptr).astype(int)
                    if an_frame >=  X1m.shape[1]:  an_frame =  X1m.shape[1] - 1
                    an_frame1 = min(np.floor(an_ptr).astype(int), X1m.shape[1] - 1)
                    an_frame2 = min(np.ceil(an_ptr).astype(int), X1m.shape[1] - 1)
                    if rest_rate > 0:
                        frac = an_ptr - np.floor(an_ptr)
                        Ym[:,i] = (1-frac)*X1m[:, an_frame1] + frac * X1m[:, an_frame2]
                    else: # extrapolate from last few frames
                        #Ym[:,i] = Ym[:,i-1] * 0.95
                        Ym[:,i] = Ym[:,i-1] * 0
                    phase_dev = get_phase_dev(Xp, phase_increment, an_frame)
                    if phase_lock:
                        p, v = get_peaks(Ym[:,i])
                        for j in range(len(v) - 1):# TODO: ignoring highest peaks for now
                            phase_env = wrap_phase(Xp[v[j]:v[j+1],an_frame] - Xp[p[j], an_frame] + phase_dev[p[j]])
                            Yp[v[j]:v[j+1], i]  = prev_syn_phase[p[j]] + phase_increment[p[j]] + phase_env
                    else:
                        phase_dev = get_phase_dev(Xp, phase_increment, an_frame)
                        Yp[:,i] = prev_syn_phase + phase_increment + phase_dev
                        Yp[:,i] = Xp[:, an_frame]
                    prev_syn_phase = Yp[:,i]
                    an_ptr = an_ptr + rest_rate
    Y = Ym * np.exp(1j * Yp)
    Y = Spectrogram(Y, X.sample_rate, window_size, hop_size)
    str_estimate = istft.process(Y)
    if stretched is None: stretched = np.zeros_like(str_estimate)
    stretched = stretched + str_estimate
write_mono(stretched, out_fname, sr)
