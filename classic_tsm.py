import numpy as np
from scipy import signal
from util import *
import stft

"""
Classic TSM methods:
    - OLA / WSOLA (OLA is WSOLA with tolerance = 0)
    - Phase vocoder with/without phase locking
Implementations ported from TSM toolbox for Matlab
"""


def wsola(x, factor, Hs = 512, window = signal.hann(1024, sym = False), tolerance = 512):
    in_size = x.shape[0]
    win_len = window.shape[0]
    win_len_half = int(np.round(win_len / 2))
    out_size = int(np.ceil(factor * in_size))
    anchor_points = np.array([[0, 0],[in_size - 1, out_size - 1]])
    syn_positions = np.arange(0, out_size + win_len_half, Hs)
    an_positions = np.round(
        np.interp(syn_positions, anchor_points[:,1], anchor_points[:,0])
    )
    an_hops =  np.concatenate(([0], an_positions[1:] - an_positions[:-1]))
    y = np.zeros((out_size + 2 * win_len))
    x = np.concatenate((
        np.zeros((win_len_half + tolerance)),
        x,
        np.zeros((int(np.ceil(1 / factor)) * win_len + tolerance))
    ))
    an_positions += tolerance
    window_norm = np.zeros(out_size + 2 * win_len)
    delay = 0
    for i in range(an_positions.size - 1):
        syn_win_range = np.arange(
            syn_positions[i],
            syn_positions[i] + win_len,
            dtype = int
        )
        an_win_range = np.arange(
            an_positions[i] + delay,
            an_positions[i] + win_len + delay,
            dtype = int
        )
        y[syn_win_range] += x[an_win_range] * window
        window_norm[syn_win_range] += window
        natural_prog = x[an_win_range + Hs]
        next_an_win_range = np.arange(
            an_positions[i + 1] - tolerance,
            an_positions[i + 1] + win_len + tolerance,
            dtype = int
        )
        next_an_win_val = x[next_an_win_range]
        if tolerance > 0:
            cros_corr = signal.convolve(
                np.flip(next_an_win_val, 0), natural_prog
            )[win_len:-win_len + 1]
            max_index = np.argmax(cros_corr)
            delay = tolerance - max_index - 1
        else:
            delay = 0
    last_an_win_range = np.arange(
        an_positions[-1] + delay,
        an_positions[-1] + win_len + delay,
        dtype = int
    )
    y[syn_positions[-1]:syn_positions[-1] + win_len] += x[last_an_win_range] * window
    window_norm[syn_positions[-1]:syn_positions[-1] + win_len] += window
    window_norm[window_norm<10**-3] = 1
    y /= window_norm
    y = y[win_len_half:out_size]
    return y


def pvoc(x, sr, factor, Hs = 512, window = signal.hann(1024, sym = False), phase_lock = False):
    in_size = x.shape[0]
    win_len = window.shape[0]
    win_len_half = int(np.round(win_len / 2))
    out_size = int(np.ceil(factor * in_size))
    anchor_points = np.array([[0, 0],[in_size - 1, out_size - 1]])
    syn_positions = np.arange(0, out_size + win_len_half, Hs)
    an_positions = np.round(
        np.interp(syn_positions, anchor_points[:,1], anchor_points[:,0])
    )
    an_hops =  np.concatenate(([0], an_positions[1:] - an_positions[:-1]))
    y = np.zeros((out_size + 2 * win_len))
    x = np.concatenate((
        np.zeros((win_len_half)),
        x,
        np.zeros((win_len + int(an_hops[1])))
    ))

    X = stft.stft(x, sr, an_positions, window, win_len)
    Y = np.zeros_like(X)
    Y[:,0] = X[:,0] #assuming columns are frames
    k = np.arange(win_len_half + 1).T
    omega = 2 * np.pi * k / win_len
    print(an_hops[1])
    print(an_hops[-1])
    for i in range(1, X.shape[1]):
        dphi = omega * an_hops[i]
        current_phase = np.angle(X[:,i])
        prev_phase = np.angle(X[:,i - 1])
        phase_inc = current_phase - prev_phase - dphi
        phase_inc = phase_inc - 2 * np.pi *np.round(phase_inc / (2*np.pi))
        ipa_sample = omega + phase_inc / an_hops[i]
        ipa_hop = ipa_sample * Hs
        syn_phase = np.angle(Y[:,i - 1])
        if not phase_lock:
            theta = syn_phase + ipa_hop - current_phase
            phasor = np.exp(1j*theta)
        else:
            p, v = get_peaks(np.abs(X[:,i]))
            theta = np.zeros_like(Y[:,i])
            for j in range(len(p)):
                theta[v[j]:v[j+1]] = syn_phase[p[j]] + ipa_hop[p[j]] - current_phase[p[j]]
            phasor = np.exp(1j*theta)
        Y[:,i] = phasor * X[:,i]
    y = stft.istft(Y, Hs, window)
    return y
