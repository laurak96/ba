import os
import pickle

import numpy as np
import pandas as pd
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from fastdtw import fastdtw
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MaxAbsScaler


def preprocess_data(data):
    """
    Preprocesses the Data. Data will be smoothed with a gaussian filter and afterwards scaled within a range
    of -1 to 1.
    :param data: 2d array of data points of the time series.
    :return: 2d array of smoothed and scaled data.
    """
    data_y = convolve((data[:, 1] - data[0, 1]), kernel=Gaussian1DKernel(stddev=50), boundary='extend')
    data_x = data[:, 0] - data[0, 0]
    data_y = data_y.reshape(-1, 1)
    scaler = MaxAbsScaler()
    data_y = scaler.fit_transform(data_y)
    data = np.column_stack((data_x, data_y))
    return data


def compute_dtw_distance(data1, data2):
    """
    Computes the dtw distances of semple data1 and candidate data2. It utilises the fast-dtw- algorithm.
    :param data1: The sample.
    :param data2: The candidate to check for similarity.
    :return: The dtw distance.
    """
    distance, path = fastdtw(data1[:, 1], data2[:, 1], radius=1, dist=euclidean)
    return distance


def process_file(curve_ref, file, dir, verbose):
    """
    Wrapper function for extracting the data out of a file using joblib and computing the dtw-distance.
    :param curve_ref: 2d array of the Sample curves data.
    :param file: Name to file of the candidate.
    :param dir: Path to the directory of the candidate files.
    :param verbose: Switch for turning verbosity on or off.
    :return: Dtw distance and name of candidate.
    """
    filename, file_extension = os.path.splitext(file)
    if file_extension == '.data':
        curve = preprocess_data((pd.read_csv(os.path.join(dir, file), sep=" ")).values)
        distance = compute_dtw_distance(curve_ref, curve)
        if verbose:
            print('Processing ' + file + ' done!')
        return distance, file


def process_distance(sample, dir, n_proc, verbose):
    """
    Wrapper function for calling process_file with joblib.
    :param sample: Path to the sample curves file.
    :param dir: Path to the directory of the candidate files.
    :param n_proc: Number of CPU-Cores. Default: -1 use all available Cores.
    :param verbose: Switch for turning verbosity on or off.
    :return: The dtw distances and names of the processed candidates.
    """
    curve_ref = preprocess_data((pd.read_csv(sample, sep=" ")).values)
    distances = Parallel(n_jobs=n_proc)(
        delayed(process_file)(curve_ref, file, dir, verbose) for file in os.listdir(dir))
    distances = np.asarray(distances)
    pickle.dump(distances, open('distances.pkl', "wb"))
    if verbose:
        print('finish')
    return distances


def find_similar_curve(sample, dir, est_dist=50, n_proc=-1, verbose=False):
    """
    Computes the most similar candidates of the sample curve with help of the dtw distance.
    :param sample: Path to the sample curves file.
    :param dir: Path to the directory of the candidate files.
    :param est_dist: Value for checking the dtw-distance against to find the most similar candidates.
    :param n_proc: Number of CPU-Cores. Default: -1 use all available Cores.
    :param verbose: Switch for turning verbosity on or off.
    :return: The most similar candidates of the sample curve.
    """
    res = process_distance(sample, dir, n_proc, verbose)
    dist = np.array(res[:, 0], dtype='float')
    files = res[:, 1]
    return files[np.where(dist < est_dist)]
