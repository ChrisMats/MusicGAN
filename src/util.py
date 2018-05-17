# -*- coding: utf-8 -*-
"""Module with audio-data processing tools.

This module contains functions that are used to load and process audio data.

"""

import numpy as np
import tensorflow as tf
from pysndfile import sndio



__author__ = "MattSt, matsou"

CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints/")

def get_checkpoint_path(network_name):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    return os.path.join(CHECKPOINT_DIR, network_name)

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

