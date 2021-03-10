import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.signal import argrelmin, savgol_filter, waveforms


def importFile(path, limit_low=None, limit_high=None):
    """Import the spectral data from a single file

    Args:
        path (str): Path to the file to be imported. 
        limit_low (int, optional): Lower limit for the spectral range. Defaults to None.
        limit_high (int, optional): Upper limit for the spectral range. Defaults to None.

    Returns:
        x (numpy.ndarray): x-values (e.g wavelengths, wavenumbers) of the imported spectrum.
        y (numpy.ndarray): y-values (e.g absorbance, transmission, intensity) of the imported spectrum.
    """

    spectrum = np.genfromtxt(path, delimiter=",")
    spectrum = np.transpose(spectrum)
    x = spectrum[0]
    y = spectrum[1]

    if limit_low is not None:
        limit_low_index = list(x).index(limit_low)
    else:
        limit_low_index = 0
        limit_low = x[0]

    if limit_high is not None:
        limit_high_index = list(x).index(limit_high)
    else:
        limit_high_index = len(x)
        limit_high = x[-1]

    x = x[limit_low_index:limit_high_index]
    y = y[limit_low_index:limit_high_index]
    return x, y


def importDirectory(path, limit_low=None, limit_high=None):
    """Import the spectral data from all files in a given directory. 

    Args:
        path (str): Path to the directory from which spectra should be imported.
        limit_low (int, optional): Lower limit for the spectral range. Defaults to None.
        limit_high (int, optional): Upper limit for the spectral range. Defaults to None.

    Returns:
        x (numpy.ndarray): x-values (e.g wavelengths, wavenumbers) of the imported spectra. Each row represents one file.
        y (numpy.ndarray): y-values (e.g absorbance, transmission, intensity) of the imported spectra. Each row represents one file
        files (lst): List of imported files, ordered numerically
    """

    if not path.endswith("/"):
        path = path + "/"

    files = os.listdir(path)
    files = [file for file in files if file.lower().endswith(".txt")]

    files = sorted(files, key=lambda s: int(s[s.find("(")+1:s.find(")")]))

    x = []
    y = []

    for file in files:
        wns, ints = importFile(path + file, limit_low, limit_high)
        x.append(wns)
        y.append(ints)
    return np.array(x), np.array(y), files


def _smooth(y, pen):

    diag = np.zeros((5, 5))
    np.fill_diagonal(diag, 1)
    middle = np.matmul(np.diff(diag, n=2, axis=0).T,
                       np.diff(diag, n=2, axis=0))
    zeros = np.zeros((2, 5))

    to_band = np.vstack((zeros, middle, zeros))
    the_band = np.diag(to_band)

    for i in range(1, 5):
        the_band = np.vstack((the_band, np.diag(to_band, -i)))

    indices = [0, 1] + [2] * (np.shape(y)[1]-4) + [3, 4]
    dd = the_band[:, indices] * (10 ** pen)
    dd[2, ] = dd[2, ] + 1

    y_smooth = solve_banded((2, 2), dd, y.T).T
    return y_smooth


def _subsample(y, lims, buckets):

    y_subs = np.zeros(buckets)
    for i in range(buckets):
        y_subs[i] = np.min(y[lims[i]:lims[i+1]])
    return y_subs


def _suppression(y_subs, buckets, its, windows):

    for i in range(its):
        w0 = windows[i]

        for j in range(1, buckets):
            v = min(j, w0, buckets-j)
            a = np.mean(y_subs[j-v:j+v+1])
            y_subs[j] = min(a, y_subs[j])

        for j in range(buckets-1, 0, -1):
            v = min(j, w0, buckets-j)
            a = np.mean(y_subs[j-v:j+v+1])
            y_subs[j] = min(a, y_subs[j])
    return y_subs


def peakFill_4S(y, pen, hwi, its, buckets):

    dims = np.shape(y)
    baseline = np.zeros(dims)

    if its != 1:
        d1 = math.log10(hwi)
        d2 = 0

        tmp = np.array(range(its-1)) * (d2 - d1) / (its - 1) + d1
        tmp = np.append(tmp, d2)
        windows = np.ceil(10**tmp).astype(int)
    else:
        windows = np.array((hwi))

    if isinstance(buckets, int):
        lims = np.linspace(0, dims[1]-1, buckets+1, dtype=int)
    else:
        lims = buckets
        buckets = len(lims)-1

    mids = np.rint(np.convolve(lims, np.ones(2), 'valid') / 2).astype(int)
    mids[0] = 0
    mids[-1] = dims[1]-1

    y_smooth = _smooth(y, pen)

    for s in range(len(y)):
        y_subs = _subsample(y_smooth[s], lims, buckets)
        y_supr = _suppression(y_subs, buckets, its, windows)
        baseline[s] = np.interp(range(dims[1]), mids, y_supr)

    y_corrected = y - baseline
    return y_corrected


def peakRecognition(y, sg_window):
    corrected_sg2 = savgol_filter(y_corrected, window_length=sg_window, polyorder=3, deriv=2)

    scores = []

    for i, row in enumerate(corrected_sg2):
        threshold = 0.05 + np.max(y[i])/30000
    #     print(i, threshold)
        peaks_tmp = argrelmin(row)[0]
        peaks_tmp = [peak for peak in peaks_tmp if row[peak] < -threshold]
        
        peak_condensing = []
        peaks_tmp2 = []
        for j in range(len(row)):
            if j in peaks_tmp:
                peak_condensing.append(j)
            if row[j] > 0 and len(peak_condensing) > 0:
                peaks_tmp2.append(int(np.mean(peak_condensing)))
                peak_condensing = []
        if len(peak_condensing) > 0:
            peaks_tmp2.append(int(np.mean(peak_condensing)))
        
        heights = [y[i, k] for k in peaks_tmp2]
        score = np.median(heights) ** len(heights)
        scores.append(score)
    
    return scores
    





if __name__ == '__main__':
    x, y, files = importDirectory('spectra/', 300, 1600)
    y_corrected = peakFill_4S(y, 0, 10, 6, 400)
    scores = peakRecognition(y_corrected, 35)

