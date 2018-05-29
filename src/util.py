# -*- coding: utf-8 -*-
"""Module with audio-data processing tools.

This module contains functions that are used to load and process audio data.

"""

import numpy as np
import tensorflow as tf
import os
from pysndfile import sndio
import pickle
from proto import *

__author__ = "MattSt, matsou"

CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints/")
RESULTS_DIR = os.path.join(os.getcwd(), "results/")
PARTIAL_RESULTS_DIR = os.path.join(os.getcwd(), "results/training_step_res/")

def get_checkpoint_path(network_name):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    return os.path.join(CHECKPOINT_DIR, network_name)

def get_results_path(name):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    return os.path.join(RESULTS_DIR, name)

def get_par_results_path(name):
    if not os.path.exists(PARTIAL_RESULTS_DIR):
        os.makedirs(PARTIAL_RESULTS_DIR)
    return os.path.join(PARTIAL_RESULTS_DIR, name)

def save_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_audio(file_path):
    """Loads audio data from wav file using pysndfile.

    Args:
        file_path: Path to a wav file.

    Returns:
        A tuple containing the samples and the sampling rate of the
        wav file, in this order.
    """
    data = sndio.read(file_path)
    sampling_rate = data[1]
    samples = np.array(data[0], dtype=np.float32)
    return samples, sampling_rate


# TODO: Implement function below
def crop_or_random_pad(audio_samples, audio_desired_length):
    """ Preprocesses audio samples by cropping or random padding.

    Given a fixed signal length, if the given signal is longer than the designated
    length then this function crops the sides of the signal and if the signal is
    smaller than that length it adds random padding until it meets the desired
    length.

    Args:
        audio_samples: A tensor which includes the samples of the signal.
        audio_desired_length: A tensor with the desired fixed size of the
            signal, given in that order.

    Returns:
        A tf.float32 with the signal after preprocessed so that it has the desired
        length.
    """
    pass
    # Get tensor with number of samples
    #sample_no = tf.size(audio_samples)
    # Get difference of samples with maximum length
    #diff = tf.subtract(sample_no, audio_desired_length)
    #abs_diff = tf.abs(diff)
    # Pad the sides symmetrically
    #padded_samples = tf.pad(sample_info, [[0, 0], [0, 0]], mode = 'CONSTANT')

def vizualise_samples(fake, true, itr=None,fs = 16384):
    fig = plt.figure()
#     plt.title('The signal')
    plt.axis('off')
    plt.subplots_adjust(hspace = 0.4)
    ax = fig.add_subplot(2,1,1)
    ax.plot(true)
    ax.set_title("True")
    ax = fig.add_subplot(2,1,2)
    ax.plot(fake)
    ax.set_title("Fake")
    save_plot('signal_it{}'.format(itr))
    plt.show()

    fig = plt.figure()
#     plt.title('The frames')
    plt.axis('off')
    plt.subplots_adjust(hspace = 0.4)
    frames_t = enframe(true, winlen=20, winshift=10, samplingrate=fs)
    frames_f = enframe(fake, winlen=20, winshift=10, samplingrate=fs)
    ax = fig.add_subplot(2,1,1)
    ax.pcolormesh(frames_t)
    ax.set_title("True")
    ax = fig.add_subplot(2,1,2)
    ax.pcolormesh(frames_f)
    ax.set_title("Fake")
    save_plot('frames_it{}'.format(itr))
    plt.show()
    # 4.2 - Apply pre-emphasis filter to frames
    fig = plt.figure()
    plt.axis('off')
#     plt.title('The pre-emphasis frames')
    plt.subplots_adjust(hspace = 0.4)
    pre_emphasis_t = preemp(frames_t, p=0.97)
    pre_emphasis_f = preemp(frames_f, p=0.97)
    ax = fig.add_subplot(2,1,1)
    ax.pcolormesh(pre_emphasis_t)
    ax = fig.add_subplot(2,1,2)
    ax.pcolormesh(pre_emphasis_f)
    save_plot('pre-emph_it{}'.format(itr))
    plt.show()
    # 4.3 - Apply hamming window to the pre-emphasized frames
    fig = plt.figure()
    plt.axis('off')
#     plt.title('The windowed frames')
    plt.subplots_adjust(hspace = 0.4)
    windowed_t = windowing(pre_emphasis_t)
    windowed_f = windowing(pre_emphasis_f)
    ax = fig.add_subplot(2,1,1)
    ax.pcolormesh(windowed_t)
    ax = fig.add_subplot(2,1,2)
    ax.pcolormesh(windowed_f)
    save_plot('window_it{}'.format(itr))
    plt.show()
    # 4.4 - Compute the power spectrum of the windowed frames
    fig = plt.figure()
    plt.axis('off')
#     plt.title('The power spectrogram of the windowed frames')
    plt.subplots_adjust(hspace = 0.4)
    power_spec_t = powerSpectrum(windowed_t, nfft=512)
    power_spec_f = powerSpectrum(windowed_f, nfft=512)
    ax = fig.add_subplot(2,1,1)
    ax.pcolormesh(power_spec_t)
    ax = fig.add_subplot(2,1,2)
    ax.pcolormesh(power_spec_f)
    save_plot('spectrogram_it{}'.format(itr))
    plt.show()

def save_plot(savename):
    savename = "plot_out" if savename is None else savename
    filepath = 'results'
    if savename is not None:
        savename += '.png'
        savepath = os.path.abspath(os.path.join(os.pardir, filepath))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        plt.savefig(os.path.join(savepath, savename), bbox_inches='tight')
        print(savename + " successfully saved to " + filepath)
