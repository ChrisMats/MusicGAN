"""

This script enframes a number of wav files in a dictionary and saves the frames.

"""

import argparse
import os
import sys
from tqdm import tqdm
from util import *
from proto import enframe2
from pysndfile.sndio import write
from scipy.signal import resample


__author__ = "MattSt"


# Default values for CLI arguments
DEFAULT_WAV_FILE_DIR = '../../Birds/WAV_files/' # Default Speech Commands dataset path
DEFAULT_SAVED_SAMPLE_DIR = '../../Birds/WAV_samples/' # Default path to save preprocessed data
DEFAUT_WINDOW_LENGTH = 16384 # Window overlap
DEFAULT_WINDOW_OVERLAP = 2048 # Window overlap
DEFAULT_RESAMPLE_FREQUENCY = None 



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Audio enframing.")
    parser.add_argument("--WAV-path", type=str, default = DEFAULT_WAV_FILE_DIR,
                        help="Path to the directory with the wav files.")
    parser.add_argument("--frames-path", type=str, default = DEFAULT_SAVED_SAMPLE_DIR,
                        help="Path to save the frames.")
    parser.add_argument("--win-length", type=str, default = DEFAUT_WINDOW_LENGTH,
                        help="Enframing window length in samples..")
    parser.add_argument("--win-overlap", type=str, default = DEFAULT_WINDOW_OVERLAP,
                        help="Enframing window overlap in samples.")
    parser.add_argument("--resample-freq", type=str, default = DEFAULT_RESAMPLE_FREQUENCY,
                        help="The sampling frequency to convert every wav to before saving..")
    return parser.parse_args()

args = get_arguments()


# Create directory for saving the pre-processed data if it does not exist
if not os.path.exists(args.WAV_path):
    os.makedirs(args.WAV_path)
    print("Created " + args.WAV_path + " directory.")

# Create directory for saving the pre-processed data if it does not exist
if not os.path.exists(args.frames_path):
    os.makedirs(args.frames_path)
    print("Created " + args.frames_path + " directory.")


# TODO: do sth so that samples with lower and high sampl. freq. don't have the same size
for filename in tqdm(os.listdir(args.WAV_path)):
    samples, sample_freq = load_audio(args.WAV_path + filename)
    resample_freq = sample_freq
    samples = samples.flatten()
    if args.resample_freq is not None:
        secs = len(samples)/sample_freq # Calculate seconds of wav audio
        sample_count = secs * float(args.resample_freq) # Calculate samples needed in the resampled signal with the desired frequency
        samples = resample(samples, int(sample_count)) # Resample
        sample_freq = args.resample_freq
        resample_freq = int(args.resample_freq)
    seq_length = args.win_length * sample_freq
    frames = enframe2(samples, args.win_length, args.win_overlap)
    # Save frames to wav files
    for idx, frame in enumerate(frames):
        write(args.frames_path + "_" + str(idx) + "_" + filename, frame, rate = resample_freq)
            
            
