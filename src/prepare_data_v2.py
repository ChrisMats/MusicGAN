"""

This script loads the Speech Commands dataset, builds the SC09 dataset (discards all samples except for
the utterances from 0 to 9), preprocesses the data and saves it to a designated folder as TFRecords. 

Note, in order to not discard the 0-9 utterances they must be in folders named 'zero', 'one', 'two', etc.
within the Speech Command dataset folder.

"""

import argparse
import os
import sys
import tensorflow as tf
from tqdm import tqdm
from util import *



__author__ = "MattSt"


# Default values for CLI arguments
DEFAULT_TFRECORD_NAME = 'train_'
DEFAULT_DATASET_PATH = '../../SpeechCommands/' # Default Speech Commands dataset path 
DEFAULT_PREPROCESSED_DATASET_PATH = '../../SC09_Preprocessed/' # Default path to save preprocessed data
DEFAULT_FIXED_SIGNAL_SIZE = 16384 # Default size to pad/crop signals to
DEFAULT_RANDOM_PADDING = False # True if random padding should be applied
DEFAULT_MINI_BATCH_SIZE = 12


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dataset preprocessing.")
    parser.add_argument("--SC-path", type=str, default = DEFAULT_DATASET_PATH,
                        help="Path to the Speech Commands dataset folder.")
    parser.add_argument("--SC09-path", type=str, default = DEFAULT_PREPROCESSED_DATASET_PATH,
                        help="Path to save the preprocessed SC09 dataset.")
    parser.add_argument("--fixed-length", type=str, default = DEFAULT_FIXED_SIGNAL_SIZE,
                        help="Signals should be cropped/padded to have this fixed length.")
    parser.add_argument("--random-padding", type=str, default = DEFAULT_RANDOM_PADDING,
                        help="Set to true if you want to apply random padding.")
    parser.add_argument("--dataset-fname", type=str, default = DEFAULT_TFRECORD_NAME,
                        help="Desired file name for the TFRecord dataset.")
    parser.add_argument("--minibatch-size", type=str, default = DEFAULT_MINI_BATCH_SIZE,
                        help="Default mini-batch size.")
    return parser.parse_args()



# Folders with 0-9 classes in speech commands
class_folders = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine']

args = get_arguments()


# Create directory for saving the pre-processed data if it does not exist
if not os.path.exists(args.SC_path):
    os.makedirs(args.SC_path)
    print("Created " + args.SC_path + " directory.") 
    
    
############################ Build the tensorflow preprocessing graph ############################
audio_file_path = tf.placeholder(tf.string, [])

# Create tf operation that loads audio and returns samples and sampling rate
audio_samples, audio_sampling_freq = tf.py_func(load_audio, [audio_file_path], [tf.float32, tf.int64],                                                name='Data_loading') 

# Constant for desired signal size after cropping or padding
fixed_signal_size = tf.constant(args.fixed_length)

# Preprocess audio
if args.random_padding:
    # Crop sides if larger than fixed length, insert random padding otherwise
    #preprocessed_samples = tf.py_func(crop_or_random_pad, [audio_samples, fixed_signal_size], tf.float32)
    pass
else:
    # Crop sides if larger than fixed length, pad the sides of the signal symmetrically if not large enough
    tensor_height = tf.constant(1)
    reshaped_samples = tf.reshape(audio_samples, [1, tf.size(audio_samples), -1])
    preprocessed_samples = tf.image.resize_image_with_crop_or_pad(reshaped_samples, tensor_height, fixed_signal_size)
    preprocessed_samples = tf.reshape(preprocessed_samples, [1, fixed_signal_size])


#################### Create tf session, preprocess and save data as TFRecords ####################
sess = tf.Session()

# Create TFRecord writer
writer = tf.python_io.TFRecordWriter(args.SC09_path + args.dataset_fname + '0.tfrecords')

sample_count = 0
saved_sample_counter = 0
minibatch_counter = 0
# For each folder in the dataset folder
for folder_name in os.listdir(args.SC_path):
    # Ommit the folder if its name does not correspond to 0-9 digits.
    if folder_name not in class_folders:
        continue
    
    folder_path = args.SC_path + folder_name + "/"
    print("Preprocessing samples from folder " + folder_path + ".")
    folder_file_count = 0
    for wav_file_name in tqdm(os.listdir(folder_path)):
        # Make ndarray with wav file path
        load_file_path = np.array(folder_path + wav_file_name)
        # Load and process the wav file using tensorflow
        preprocessed_data = sess.run(preprocessed_samples, feed_dict = {audio_file_path: load_file_path})
        
        # Create a feature 
        feature = {'signal': _bytes_feature(tf.compat.as_bytes(preprocessed_data.tostring())),
                    'label': _bytes_feature(tf.compat.as_bytes(folder_name))}
        # Create example protocol buffer
        exmple_protocol_buff = tf.train.Example(features=tf.train.Features(feature = feature))
        
        
        # If a full minibatch has been saved
        if(saved_sample_counter >= args.minibatch_size):
            # Close previous TFRecord writer
            writer.close()
            sys.stdout.flush()
            # Increment mini-batch counter
            minibatch_counter += 1
            # Create TFRecord writer
            writer = tf.python_io.TFRecordWriter(args.SC09_path + args.dataset_fname + str(minibatch_counter) + '.tfrecords')
            # Reset sample counter
            saved_sample_counter = 0
            
        
        # Serialize to string and write to file
        writer.write(exmple_protocol_buff.SerializeToString())
        # Increment saved sample counter
        saved_sample_counter += 1 
        # Increase count of current class files by 1
        folder_file_count += 1
    
    # Increase count of samples processed
    sample_count += folder_file_count
    
    print(str(folder_file_count) + " samples from folder " + folder_path + \
          " have been preprocessed and saved in " + args.SC09_path + ".\n")
    
# Close TFRecord writer
writer.close()
sys.stdout.flush()
print("Loaded, preprocessed and saved " + str(sample_count) + " wav files as TF records.")

