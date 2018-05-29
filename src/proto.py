# DT2119, Lab 1 Feature Extraction

import matplotlib.pylab as plt
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import lfilter, hamming
from scipy.spatial.distance import euclidean

from tools import lifter, trfbank

import pdb
# Function given by the exercise ----------------------------------

def mfcc(samples, winlen=20, winshift=10, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift, samplingrate)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------

def enframe(samples, winlen=20, winshift=10, samplingrate=20000):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    winlen = int(winlen * 0.001 * samplingrate)
    winshift = int(winshift * 0.001 * samplingrate)

    frames = []

    for step in range(0, len(samples), winshift):

        if step + winlen < len(samples):
            frames.append(samples[step:step + winlen])

    return np.asarray(frames)

# Alternative enframe implementation
def enframe2(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.
    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    A = np.array(samples[0:winlen].reshape((1, winlen)))
    stepsize = winlen - winshift
    for i in range(stepsize, len(samples) - winlen, stepsize):
        A = np.vstack((A, samples[i:i+winlen].reshape((1, winlen))))
    return A


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    return lfilter([1, -p], [1], input)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    return input * hamming(np.shape(input)[1], sym=False)


def powerSpectrum(input, nfft=512):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    return abs(fft(input, n=nfft))


def logMelSpectrum(input, samplingrate=20000, plotfilers=False):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    filter_bank = trfbank(samplingrate, np.shape(input)[1])

    if plotfilers:
        plt.plot(filter_bank)
        plt.xlabel("spectrum length")
        plt.ylabel("filter amplitudes")
        plt.title("Mel filters in linear frequency scale")
        plt.show()

    return np.log(input.dot(filter_bank.T))


def cepstrum(input, nceps=13):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return dct(input)[:, 0:nceps]


def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        global_d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        local_d: local distance between frames from x and y (NxM matrix)
        acc_d: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    local_d = dist(x, y)
    global_d = np.zeros(local_d.shape)
    global_d[0][0] = float("inf")

    for i in range(1, global_d.shape[0]):
        for j in range(1, global_d.shape[1]):
            global_d[i][j] = local_d[i][j] + min(global_d[i - 1, j],  # insertion
                                           global_d[i, j - 1],  # deletion
                                           global_d[i - 1, j - 1])  # match
    global_d = global_d[x.shape[0] - 1, y.shape[0] - 1] / (x.shape[0] + y.shape[0])
    return global_d, local_d

def efc_dist(x, y):
    """ Compute the Euclidean distance between 2 utterances"""

    dist = np.zeros((x.shape[0], y.shape[0]))
    for i, value_1 in enumerate(x):
        for j, value_2 in enumerate(y):
            dist[i][j] = euclidean(value_1, value_2)
    return dist
