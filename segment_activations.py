
import numpy as np
from collections import namedtuple
from scipy.ndimage.filters import gaussian_filter1d

"""
Segment NMF activations.
Phase 1: threshold into on-off regions (get_segments)
Phase 2: separate into Events composed of:
    - attack transient
    - active region
    - silence region
A part is a list of 3 lists, corresponding to this model, each of the lists contains start-end indices
"""

Event = namedtuple('Event', 'start end n_transient n_active n_silence')

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

def find_segments(act, min_segment = 3, t1 = 0.5, t2 = 0):

    smooth_act = act
    th1 = np.mean(smooth_act) + t1*np.std(smooth_act)
    th2 = np.mean(smooth_act)+ t2*np.std(smooth_act)
    above = np.where(smooth_act > th1)
    initial_segments = np.split(above[0], np.where(np.diff(above[0]) != 1)[0] + 1)
    initial_segments = [s for s in initial_segments if len(s) > min_segment]
    th2_crossings = np.where(np.diff(np.signbit(smooth_act - th2)))[0]
    final_segments = []
    for s in initial_segments:
        closest_crossings = np.where(th2_crossings - s[0] > 0)[0]
        if len(closest_crossings) == 0:
            next_crossing = s[-1]
        else:
            next_crossing = th2_crossings[closest_crossings[0]]
        if len(final_segments) == 0 or next_crossing - 1 != final_segments[-1][-1]:
            final_segments.append(np.arange(s[0], next_crossing))
    return final_segments

def find_transients(seg, sig, t3 = 0.1):
    dif = np.diff(np.concatenate(([0],sig)))
    th = np.mean(dif) + t3 * np.std(dif)
    peaks = detect_peaks(dif[seg])
    return [seg[0] + p - 1 for p in peaks if dif[seg][p]>th]

def make_event(transient, segment, end):
    if transient[1] > end: transient[1] = end
    if segment[1] < transient[1]: segment[1] = transient[1]
    if segment[1] > end: segment[1] = end
    return Event(transient[0], end, transient[1] - transient[0], segment[1] - transient[1], end - segment[1])

def make_analysis_events(activation, t1 = 0.5, t2 = 0, t3 = 0.1):
    transient_dur = 4
    n_frames = activation.shape[0]
    active_segments = find_segments(activation,3, t1, t2)
    transients = [find_transients(seg, activation, t3) for seg in active_segments]
    events = [Event(0, active_segments[0][0], 0, 0, active_segments[0][0])]
    for i in range(len(active_segments)):
        if i == len(active_segments) - 1:
            end = n_frames
        else:
            if len(transients[i + 1]) == 0:
                end = active_segments[i+1][0]
            else:
                end = transients[i + 1][0]
        seg = [active_segments[0], active_segments[i][-1]]
        if (len(transients[i]) == 0):
            transient = [active_segments[i][0], active_segments[i][0]]
            events.append(make_event(transient, seg, end))
        elif (len(transients[i]) == 1):
            events.append(make_event(
                [transients[i][0], transients[i][0] + transient_dur],
                seg, end)
            )
        else:
            for j in range(len(transients[i]) - 1):
                n_active = transients[i][j + 1] - transients[i][j] - transient_dur
                ev = Event(transients[i][j], transients[i][j + 1], transient_dur, n_active, 0)
                events.append(ev)
            last_tr = transients[i][-1]
            events.append(make_event(
                [last_tr, last_tr + transient_dur],
                [last_tr + transient_dur, seg[1]],
                end)
            )
    return events

def make_synthesis_events(analysis_events, factor, lock_active = False):
    n_in_frames = analysis_events[-1].end
    n_out_frames = int(np.round(n_in_frames * factor))
    start_pos = [e.start for e in analysis_events]
    new_pos = np.round(np.interp(start_pos, [0, n_in_frames], [0, n_out_frames])).astype(int)
    new_events = []
    for i in range(len(analysis_events)):
        src = analysis_events[i]
        new_start = new_pos[i]
        if i == len(new_pos) - 1:
            new_end = n_out_frames
        else:
            new_end = new_pos[i + 1]
        if lock_active and src.n_silence > 0:
            new_active_length = src.n_active
            if new_start + src.n_transient + new_active_length > new_end:
                new_active_length = new_end - src.n_transient - new_start
        else:
            new_active_length =  int(np.round(((src.n_transient + src.n_active) * factor) - src.n_transient))

        new_event = make_event(
            [new_start, new_start + src.n_transient],
            [new_start, new_start + new_active_length],
            new_end
        )
        new_events.append(new_event)
    return new_events
