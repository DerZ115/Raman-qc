import argparse
import math
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import solve_banded
from scipy.signal import argrelmin, savgol_filter


# parser = argparse.ArgumentParser()

# parser.add_argument()

path = "spectra3"
limit_low = 300
limit_high = 1600
penalty = 1
buckets = 400
half_width = 10
iterations = 6
sg_window = 35


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
        x0, y0 = importFile(path + file, limit_low, limit_high)
        x.append(x0)
        y.append(y0)
    return np.array(x), np.array(y), files


def _prep_smoothing_matrix(pen):
    """Generate the smoothing matrix for the Whittaker Smoother.

    Args:
        pen (int): Penalty to the second derivative used in the smoothing process. Higher --> Stronger Smoothing

    Returns:
        sparse_matrix (np.ndarray): Matrix used in the smoothing process. 
    """
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
    sparse_matrix = the_band[:, indices] * (10 ** pen)
    sparse_matrix[2, ] = sparse_matrix[2, ] + 1
    return sparse_matrix


def _prep_buckets(buckets, len_x):
    """Calculate the position of the buckets for the subsampling step.

    Args:
        buckets (int or list/1D-array): Either the number of buckets, or a list of bucket positions.
        len_x (int): Number of data points per spectrum.

    Returns:
        lims (np.ndarray): Array of the bucket boundaries.
        mids (np.ndarray): Central position of every bucket.
    """
    if isinstance(buckets, int):
        lims = np.linspace(0, len_x-1, buckets+1, dtype=int)
    else:
        lims = buckets
        buckets = len(lims)-1

    mids = np.rint(np.convolve(lims, np.ones(2), 'valid') / 2).astype(int)
    mids[0] = 0
    mids[-1] = len_x - 1

    return lims, mids


def _prep_window(hwi, its):
    """Calculate the suppression window half-width for each iteration.

    Args:
        hwi (int): Initial half-width of the suppression window.
        its (int): Number of iterations for the main peak suppression loop.

    Returns:
        windows (np.ndarray): array of the exponentially decreasing window half-widths.
    """
    if its != 1:
        d1 = math.log10(hwi)
        d2 = 0

        tmp = np.array(range(its-1)) * (d2 - d1) / (its - 1) + d1
        tmp = np.append(tmp, d2)
        windows = np.ceil(10**tmp).astype(int)
    else:
        windows = np.array((hwi))
    return windows


def smooth_whittaker(y, dd):
    """Smooth data with a Whittaker Smoother.

    Args:
        y (np.ndarray): The data to be smoothed, with each row representing one dataset (spectrum, etc).
        dd (np.ndarray): Smoothing matrix for the whittaker algorithm.

    Returns:
        y_smooth (np.ndarray): The smoothed data.
    """

    y_smooth = solve_banded((2, 2), dd, y.T).T
    return y_smooth


def subsample(y, lims):
    """Split the data into equally sized buckets and return the minimum of each bucket. 

    Args:
        y (np.ndarray): The data to be subsampled.
        lims (np.ndarray): The boundaries of the buckets.
        buckets ([type]): The number of buckets.

    Returns:
        y_subs (np.ndarray): The minimum value for each bucket. 
    """
    buckets = len(lims) - 1
    y_subs = np.zeros(buckets)
    for i in range(buckets):
        y_subs[i] = np.min(y[lims[i]:lims[i+1]])
    return y_subs


def suppression(y_subs, buckets, its, windows):
    """Suppress peaks in the data, once each going forward and backward through the data. 

    Args:
        y_subs (np.ndarray): The subsampled data
        buckets (int): The number of buckets used for subsampling
        its (int): The number of iterations for peak suppression
        windows (np.ndarray): The exponentially decreasing window widths for each iteration.

    Returns:
        y_subs (np.ndarray): The subsampled data with peaks suppressed
    """

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
    """[summary]

    Args:
        y ([type]): [description]
        pen ([type]): [description]
        hwi ([type]): [description]
        its ([type]): [description]
        buckets ([type]): [description]

    Returns:
        [type]: [description]
    """

    dims = np.shape(y)
    baseline = np.zeros(dims)

    smooth_matrix = _prep_smoothing_matrix(pen)
    lims, mids = _prep_buckets(buckets, dims[1])
    windows = _prep_window(hwi, its)

    y_smooth = smooth_whittaker(y, smooth_matrix)

    for s in range(len(y)):
        y_subs = subsample(y_smooth[s], lims)
        y_supr = suppression(y_subs, buckets, its, windows)
        baseline[s] = np.interp(range(dims[1]), mids, y_supr)

    y_corrected = y - baseline
    return y_corrected


def peakRecognition(y, sg_window):
    """[summary]

    Args:
        y ([type]): [description]
        sg_window ([type]): [description]

    Returns:
        [type]: [description]
    """

    corrected_sg2 = savgol_filter(
        y, window_length=sg_window, polyorder=3, deriv=2)

    scores = []
    med_heights_all = []
    n_peaks_all = []

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
        med_heights = np.median(heights)
        n_peaks = len(heights)
        score = med_heights ** (n_peaks/10)
        scores.append(score)
        med_heights_all.append(med_heights)
        n_peaks_all.append(n_peaks)

    return scores, med_heights_all, n_peaks_all


def export_sorted(path, files, scores, x, y_corr):
    dest_raw = os.path.join(path, "sorted_spectra")
    dest_corr = os.path.join(path, "baseline_corrected")

    if not os.path.exists(dest_raw):
        os.mkdir(dest_raw)
    if not os.path.exists(dest_corr):
        os.mkdir(dest_corr)

    files_sorted = [item[1] for item in sorted(zip(scores, files))]
    files_sorted.reverse()

    for i in range(len(files_sorted)):
        file = files_sorted[i]
        src_file = os.path.join(path, file)
        dest_raw_file = os.path.join(dest_raw, file)
        new_file = str(i+1) + "_" + file
        new_file_raw = os.path.join(dest_raw, new_file)
        i_orig = files.index(file)

        if os.path.exists(new_file_raw):
            if os.path.samefile(src_file, new_file_raw):
                continue
            os.remove(new_file_raw)

        shutil.copy(src_file, dest_raw)
        os.rename(dest_raw_file, new_file_raw)

        dest_corr_file = os.path.join(dest_corr, new_file)
        with open(dest_corr_file, "w+") as f:
            for j in range(len(x[i_orig])):
                f.write(str(x[i_orig, j]) + "," + str(y_corr[i_orig, j]))


if __name__ == '__main__':
    start_time = time.perf_counter()
    x, y, files = importDirectory(path, limit_low, limit_high)
    y_corrected = peakFill_4S(y, penalty, half_width, iterations, buckets)
    scores, heights, peaks = peakRecognition(y_corrected, sg_window)

    export_sorted(path, files, scores, x, y_corrected)

    end_time = time.perf_counter()

    print(f"Analyzed {len(files)} files in {round(end_time-start_time, 2)} seconds.")

    sns.set(style="ticks")

    fig, ((ax_box1, ax_box2),(ax_hist1, ax_hist2)) = plt.subplots(2, 2, sharex="col", gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(x=peaks, ax=ax_box1)
    sns.boxplot(x=heights, ax=ax_box2)
    sns.histplot(peaks, ax=ax_hist1)
    sns.histplot(heights, ax=ax_hist2)
    
    ax_box1.set(yticks=[])
    ax_box2.set(yticks=[])
    sns.despine(ax=ax_hist1)
    sns.despine(ax=ax_hist2)
    sns.despine(ax=ax_box1, left=True)
    sns.despine(ax=ax_box2, left=True)

    ax_hist1.set_xlabel("Number of Peaks")
    ax_hist2.set_xlabel("Median Peak Height")

    ax_box1.tick_params(axis="x", labelbottom=True)
    ax_box2.tick_params(axis="x", labelbottom=True)

    plt.show()