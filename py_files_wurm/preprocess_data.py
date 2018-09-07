import os

import numpy as np
import pandas as pd
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import butter, lfilter, filtfilt


def normalize(data, lower_bound=0, upper_bound=1):
    """
    Range-based normalization of the data in range from lower bound to upper bound.
    Default is the range from 0 to 1.
    :param data: 1d array of data values.
    :param lower_bound: Lower bound in the interval to normalize.
    :param upper_bound: Upper bound in the interval to normalize.
    :return: Normalized data.
    """
    d_min = np.min(data)
    d_max = np.max(data)
    return (upper_bound - lower_bound) * (data - d_min) / (d_max - d_min) + lower_bound


def standardize(data):
    """
    Standardization of the data.
    :param data: 1d array of data values.
    :return: Standardized data.
    """
    std_d = np.std(data)
    d_mean = np.mean(data)
    return (data - d_mean) / std_d


def fbfilter(data, cutoff, sample_freq=100, order=5):
    """
    A forward, backward low pass filter.
    :param data: Data to be filtered.
    :param cutoff: Frequency to cut off.
    :param sample_freq: Sample frequency.
    :param order: The order.
    :return: Filtered data.
    """
    b, a = butter(order, cutoff / (0.5 * sample_freq), btype='low', analog=False)
    return filtfilt(b, a, data)


def lowpass(data, cutoff, sample_freq=100, order=5):
    """
    A lowpass filter. Provided by Prof. Wismueller
    :param data: Data to be filtered.
    :param cutoff: Frequency to cut off.
    :param sample_freq: Sample frequency.
    :param order: The order.
    :return: Filtered data.
    """
    b, a = butter(order, cutoff / (0.5 * sample_freq), btype='low', analog=False)
    return lfilter(b, a, data)


def find_min_time(data):
    """
    Finds the lowes time value in a DataFrame.
    :param data: The DataFrame.
    :return: The lowest time vales.
    """
    res = data[0]['time'].iloc[0]
    for d in data:
        if d['time'].iloc[0] < res:
            res = d['time'].iloc[0]
    return res


def find_max_time(data):
    """
    Finds the biggest time value in a DataFrame.
    :param data: The DataFrame.
    :return: The biggest time value.
    """
    res = 0
    for d in data:
        if d['time'].iloc[-1] > res:
            res = d['time'].iloc[-1]
    return res


def extract_data_search(path, sep=' ', freq=1, fil=True, cutoff=1, sample_freq=100, order=5):
    """
    Extracts time series out of .data-files and applies interpolation and a forward backward low pass filter on it.
    :param path: Folder which contains the data files
    :param sep: separator used in the files, default is a whitespace
    :param freq: New sample frequency
    :param fil: Switch for enabling the fb filter
    :param cutoff: Frequency to filter
    :param sample_freq: Original sample frequency
    :param order: The order.
    :return: filtered data points and corresponding solpe values.
    """
    data = []
    file_names = []
    for file in os.listdir(path):
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.data':
            df = pd.read_csv(os.path.join((path + file)), sep=sep, header=None, names=['time', 'value'])
            data.append(df)
            file_names.append(filename)
    min_time = find_min_time(data)
    max_time = find_max_time(data)
    periods = (max_time - min_time) / freq
    r = pd.date_range(start='1970-01-01 01:00:00', periods=periods, freq=(str(freq) + 'ms'))
    points = pd.DataFrame(index=r)
    slope = pd.DataFrame(index=r)
    for i in range(0, len(data)):
        f = InterpolatedUnivariateSpline(data[i]['time'].values, data[i]['value'].values)
        if fil:
            filtered = fbfilter(f(data[i]['time']), cutoff, sample_freq, order)
            f = InterpolatedUnivariateSpline(data[i]['time'], filtered)
        dx = f.derivative()
        points[file_names[i]] = f(np.arange(int(min_time), int(max_time), freq))
        slope[file_names[i]] = dx(np.arange(int(min_time), int(max_time), freq))
    points = points.T
    slope = slope.T
    points.sort_index(inplace=True)
    slope.sort_index(inplace=True)
    return points, slope


def smooth_and_norm(values, norm_func, smooth_value):
    """
    Smooth and normalize or standardize values.
    :param values: The values to process.
    :param norm_func: Function for normalization.
    :param smooth_value: Controls how strong the curve is smoothed. Default is 10 (not much smoothing).
    :return: Smoothed and normalized values.
    """
    y = convolve(values, kernel=Gaussian1DKernel(stddev=smooth_value), boundary='extend')
    if norm_func is not None:
        y = norm_func(y)
        y = np.nan_to_num(y)
    return y


def interpolate_and_differantiate_spline(values, time, start, stop, step):
    """
    Interpolaties and differentiates values.
    :param values: Y values to process.
    :param time: X values to porcess.
    :param start: Start point of the range of the range to extract.
    :param stop: Endpoint point of the range to extract.
    :param step: Step width.
    :return: Processed values.
    """
    f = InterpolatedUnivariateSpline(time, values)
    dx = f.derivative()
    x = np.arange(start, stop, step)
    y = np.apply_along_axis(f, 0, x)
    s = np.apply_along_axis(dx, 0, x)
    return y, s


def interpolate_spline(values, time, start, stop, step):
    """
    Interpolates a spline and returns values of the spline function.
    :param values: Y values with which the spline is interpolated (as ndarray).
    :param time: X values with which the spline is interpolated (as ndarray).
    :param start: Start point of the range of the range to extract.
    :param stop: Endpoint point of the range to extract.
    :param step: Step width.
    :return:
    """
    f = InterpolatedUnivariateSpline(time, values)
    x = np.arange(start, stop, step)
    y = np.apply_along_axis(f, 0, x)
    return y


def interpolate_spline_without_range(values, time):
    """
    Interpolates a spline and returns values of the spline function.
    :param values: Y values with which the spline is interpolated (as ndarray).
    :param time: X values with which the spline is interpolated (as ndarray).
    :return: Interpolated values.
    """
    f = InterpolatedUnivariateSpline(time, values)
    y = np.apply_along_axis(f, 0, time)
    return y


def differentiate_spline_without_range(values, time):
    """
    Build the first derivative and compute slope values.
    :param values: Discrete values as ndarray.
    :param time: Discrete time as ndarray.
    :return: Slope values as ndarray.
    """
    f = InterpolatedUnivariateSpline(time, values)
    dx = f.derivative()
    s = np.apply_along_axis(dx, 0, time)
    return s


def differentiate_spline(values, time, start, stop, step):
    """
    Build the first derivative and compute slope values.
    :param values: Discrete values as ndarray.
    :param time: Discrete time as ndarray.
    :param start: Start of the time period.
    :param stop: End of the time period.
    :param step: Step width of the time period.
    :return: Slope values as ndarray.
    """
    f = InterpolatedUnivariateSpline(time, values)
    dx = f.derivative()
    x = np.arange(start, stop, step)
    s = np.apply_along_axis(dx, 0, x)
    return s


def extract_data_from_dataframe(path, start, stop, step, smooth=True, smooth_value=10,
                                norm_func=lambda x: standardize(x), imp_func=None):
    """
    Extracts the data out of a R DataFrame stored in a csv file.
    :param path: Path to the csv file.
    :param start: Start point of the range to extract.
    :param stop: Endpoint point of the range to extract.
    :param step: Step width of the points to extract (e.g. 2 mean every second point).
    :param smooth: Switch for smoothing the data points (enable is default).
    :param smooth_value: Controls how strong the curve is smoothed. Default is 10 (not much smoothing).
    :param norm_func: Function which is used for normalization or standardization.
    :param imp_func: Used for determining the data format to be imported.
    :return: Sata points and slope values.
    """
    periods = (stop - start) / step
    range = pd.date_range(start='1970-01-01 01:00:00', periods=periods, freq=(str(step) + 'ms'))
    if imp_func == 'r':
        df = pd.read_csv(path)
        time = np.arange(0, df.values.shape[0], 1)
        del df['Unnamed: 0']
    elif imp_func == 'ex':
        df = pd.read_csv(path, sep=' ')
        time = np.arange(0, df.values.shape[0], 1)
        del df['time']
    else:
        raise NotImplementedError
    if imp_func == 'r' or imp_func == 'ex':
        columns = list(df)
        df = df.T
        values = df.values
        if smooth:
            values = np.apply_along_axis(smooth_and_norm, 1, values, norm_func, smooth_value)
        p_values = np.apply_along_axis(interpolate_spline, 1, values, time, start, stop, step)
        s_values = np.apply_along_axis(differentiate_spline, 1, p_values,
                                       np.arange(start, start + p_values.shape[1], 1), start,
                                       stop, step)
        points = pd.DataFrame(p_values.T, index=range, columns=columns)
        slope = pd.DataFrame(s_values.T, index=range, columns=columns)
        points = points.T
        slope = slope.T
    else:
        raise NotImplementedError
    return points, slope


def extract_data(path, start, stop, step, sep=' ', smooth=True, smooth_value=10, norm_func=None):
    """
    Extract a given time period out of all available data-files under the path and computes the slope of each point.
    :param path: Path to the data files.
    :param start: Start of the time period in ms.
    :param stop: End of the timtest2[0]=0e period in ms.
    :param step: Step width for the extracted data points.
    :param sep: String to use as separator default is a single space.
    :param smooth: Turns soothing of the curve on or off. Default is True (smoothing is activated).
    :param smooth_value: Controls how strong the curve is smoothed. Default is 10 (not much smoothing).
    :param norm_func: Function which is used for normalization or standardization.
    :return: Data points and slope values
    """
    periods = (stop - start) / step
    range = pd.date_range(start='1970-01-01 01:00:00', periods=periods, freq=(str(step) + 'ms'))
    points = pd.DataFrame(index=range)
    slope = pd.DataFrame(index=range)
    for file in os.listdir(path):
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.data':
            df = pd.read_csv(os.path.join((path + file)), sep=sep, header=None)
            df2 = pd.DataFrame(columns=['time', 'value'])
            if df.shape[0] < df.shape[1]:
                df2['time'] = np.arange(0, df.T.shape[0], 1)
                df2['value'] = df.T.values
            if df.shape[0] >= df.shape[1]:
                df2['time'] = df.values[:, 0]
                df2['value'] = df.values[:, 1]
            if smooth:
                df2['value'] = smooth_and_norm(df2['value'].values, norm_func, smooth_value)
            y, s = interpolate_and_differantiate_spline(df2['value'], df2['time'], start, stop, step)
            points[filename] = y
            slope[filename] = s
    points = points.T
    slope = slope.T
    points.sort_index(inplace=True)
    slope.sort_index(inplace=True)
    return points, slope


def extract_windows(data, size, lap=0, label=0):
    """
    Applys a sliding window over the data to extract slices.
    Slices can be lapped.
    :param data: DataFrame of which the algorithm is applied.
    :param size: Window size.
    :param lap: Lap left and right of windows size.
    :param label: Position at which a failure in the data could be, for labeling.
    :return: DataFrame with the windows, labels of the window.
    """
    values = data.values
    max_size = values.shape[1]
    offset = max_size % (size + lap)
    cor_size = max_size - offset
    col = data.index.values
    df = pd.DataFrame()
    labels = []
    for j in range(0, int(cor_size / size)):
        for i in range(0, data.shape[0]):
            start = j * (size + lap)
            stop = start + (size + 2 * lap)
            window = (values[i][start:stop])
            if window.size == size + 2 * lap:
                if label <= stop:
                    labels.append(0)
                else:
                    labels.append(1)
                df[str(col[i]) + str(j)] = window
    df = df.T
    df['label'] = labels
    return df


if __name__ == "__main__":
    # point, slope = extract_data('../../data/Sensors/correlate/', 100000, 150000, 100)
    # point, slope = extract_data('../data/Simulation/data/', 0, 2667, 1,sep='\t', smooth=True, norm_func=normalize)
    # data = extract_data('../data/Simulation/data/', 0, 2557, 1, sep='\t', smooth_value=1, smooth=True,
    #                     norm_func=normalize)[0]
    # print(point)
    # print(slope)
    # data.T.to_csv('tester.csv')
    # point, slope = extract_data_from_dataframe('../../data/Simulation/HEV/HEVData_08.09.2017.csv',0,4001,1, imp_func='ex')
    test = np.arange(21)
    test2 = np.arange(21)
    test3 = np.column_stack((test, test2))
    data = test3.T
    extract_windows(data, 3, 2, 12)
