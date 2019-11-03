import math
import os
import random
import sys
import numpy as np
import copy

from absl import app
from absl import flags
import tensorflow as tf
import pbtxthelper as pbh
#
# from google.cloud import storage

flags.DEFINE_string(
    'raw_data_dir', 'output', 'Directory path for raw yt8m dataset. '
    'Should classes subdirectories inside it.')

flags.DEFINE_string(
    'output_dir', 'output_tf', 'Directory path for tfrecords yt8m dataset. '
    'will have train and validation subdirectories inside it.')

flags.DEFINE_string(
    'pbtxt', 'label.pbtxt', 'path to write pbtxt file'
    'path to write pbtxt file')

flags.DEFINE_float(
    'val_ratio', 0.2, 'validation ratio to split dataset'
    'default is 0.2')

flags.DEFINE_integer(
    'top_n', 15, 'top n number of classes'
    'default is 15')



FLAGS = flags.FLAGS

TRAINING_SHARDS = 10
VALIDATION_SHARDS = 10

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'


def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, height, width):
    """Build an Example proto for an example.

    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
    Returns:
    Example proto
    """
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset.encode()),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename).encode()),
      'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file.

    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, output_file, filenames, synsets, labels):
    """Processes and saves list of images as TFRecords.

    Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    output_file: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: map of string to integer; id for all synset labels
    """
    writer = tf.python_io.TFRecordWriter(output_file)

    for filename, synset in zip(filenames, synsets):
        image_buffer, height, width = _process_image(filename, coder)
        label = labels[synset]
        example = _convert_to_example(filename, image_buffer, label,
                                  synset, height, width)
        writer.write(example.SerializeToString())

    writer.close()


def _process_dataset(filenames, synsets, labels, output_directory, prefix,
                     num_shards):
    """Processes and saves list of images as TFRecords.

    Args:
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: map of string to integer; id for all synset labels
    output_directory: path where output files should be created
    prefix: string; prefix for each file
    num_shards: number of chucks to split the filenames into

    Returns:
    files: list of tf-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames) / num_shards))
    coder = ImageCoder()

    files = []

    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
        chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_image_files_batch(coder, output_file, chunk_files,
                               chunk_synsets, labels)
        tf.logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files


def convert_to_tf_records(raw_data_dir):
    """Convert the Imagenet dataset into TF-Record dumps."""

    # Shuffle training records to ensure we are distributing classes
    # across the batches.
    random.seed(0)
    def make_shuffle_idx(n):
        order = list(range(n))
        random.shuffle(order)
        return order

    # Glob all the training files
    all_files = tf.io.gfile.glob(
        os.path.join(raw_data_dir, '*', '*.jpg'))

    # Get training file synset labels from the directory name
    all_synsets = [os.path.basename(os.path.dirname(f)) for f in all_files]

    # distribute in train and val
    unique_synsets = list(set(all_synsets))
    assert len(unique_synsets) > FLAGS.top_n 
    np_ts = np.array(all_synsets)
    np_tf = np.array(all_files)
    tv_dict_all = {}
    tot_files_d = {}
    cls_sel_no = []

    for o_class in unique_synsets:
        selected_mask = (np_ts == o_class)
        tot_img = sum(selected_mask)
        # val_img = int(tot_img * FLAGS.val_ratio)
        # train_img = tot_img - val_img
        cl_files = list(np_tf[selected_mask])
        random.shuffle(cl_files)
        tot_files_d[o_class] = copy.deepcopy(cl_files)
        tv_dict_all[o_class] = {'total': tot_img}
        cls_sel_no.append(tot_img)

    cls_sel_no.sort(reverse=True)
    print(cls_sel_no)
    selected_files_per_cat = cls_sel_no[FLAGS.top_n-1]
    val_no = int(selected_files_per_cat * FLAGS.val_ratio)
    train_no = selected_files_per_cat - val_no
    print('total : {0}, train : {1}, val : {2}'.format(selected_files_per_cat, train_no, val_no))
    training_files = []
    validation_files = []
    tv_dict = {}
    for key, value in tv_dict_all.items():
    	if value['total'] >= selected_files_per_cat:
    		cat_train = tot_files_d[key][:train_no]
    		cat_val = tot_files_d[key][train_no:selected_files_per_cat]
    		new_val = {'total': len(cat_train) + len(cat_val), 'val' : len(cat_val), 'train' : len(cat_train)}
    		tv_dict[key] = new_val
    		training_files = training_files + cat_train
    		validation_files = validation_files + cat_val

    print(tv_dict)
    print('------------------------------------------------------')
    training_synsets = [os.path.basename(os.path.dirname(f)) for f in training_files]
    print(len(training_files))
    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]
    training_synsets = [training_synsets[i] for i in training_shuffle_idx]


    validation_synsets = [os.path.basename(os.path.dirname(f)) for f in validation_files]
    # Create unique ids for all synsets
    labels = {v: k + 1 for k, v in enumerate(
        sorted(set(validation_synsets + training_synsets)))}

    pbh.dictToPbtxt(labels, FLAGS.pbtxt)
    # sys.exit(1)
    # Create training data
    tf.logging.info('Processing the training data.')
    training_records = _process_dataset(
        training_files, training_synsets, labels,
        os.path.join(FLAGS.output_dir, TRAINING_DIRECTORY),
        TRAINING_DIRECTORY, TRAINING_SHARDS)

    # Create validation data
    tf.logging.info('Processing the validation data.')
    validation_records = _process_dataset(
        validation_files, validation_synsets, labels,
        os.path.join(FLAGS.output_dir, VALIDATION_DIRECTORY),
        VALIDATION_DIRECTORY, VALIDATION_SHARDS)

    return training_records, validation_records


def main(argv):  # pylint: disable=unused-argument
    tf.logging.set_verbosity(tf.logging.INFO)

    # if FLAGS.gcs_upload and FLAGS.project is None:
    #   raise ValueError('GCS Project must be provided.')

    # if FLAGS.gcs_upload and FLAGS.gcs_output_path is None:
    #   raise ValueError('GCS output path must be provided.')
    # elif FLAGS.gcs_upload and not FLAGS.gcs_output_path.startswith('gs://'):
    #   raise ValueError('GCS output path must start with gs://')

    if FLAGS.output_dir is None:
        raise ValueError('output directory path must be provided.')

    if FLAGS.raw_data_dir is None:
        raise ValueError('raw_data directory path must be provided.')

    # Download the dataset if it is not present locally
    raw_data_dir = FLAGS.raw_data_dir

    # Convert the raw data into tf-records
    _, _ = convert_to_tf_records(raw_data_dir)
    print('--END--')


if __name__ == '__main__':
    app.run(main)
