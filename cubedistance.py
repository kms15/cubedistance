#!/usr/bin/python3
"""cubedistance.py - find the average distance between random points in a
    hypercube using a Monte Carlo approach.

By Kendrick Shaw 2021 -- MIT License

Usage:
  cubedistance.py [-d <max_dim>] [-p <max_power>] [-r <num_samples>]
      [-n] [-b <batch_size>] [-f <precision>]
  cubedistance.py (-h | --help)

Options:
  -h --help         Show this screen
  -d <max_dim>      Maximum hypercube dimensions [default: 100]
  -p <max_power>    Maximum power of the p-norm [default: 3]
  -r <num_samples>  Number of Monte Carlo samples for each entry
                    [default: 1000000]
  -n                Normalize to the longest diagonal
  -b <batch_size>   The maximum number of parallel values computed in each
                    batch, equal to the number of samples in the batch times
                    max_dim and max_power. [default: 1000000]
  -f <precision>    Use the specified floating point precision; valid options
                    are half, single, and double. Note that many devices do
                    not support double or half precision [default: double]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide tensorflow startup messages
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow import math as tfm
from docopt import docopt

@tf.function
def sample_batch(n_samples, max_dim, max_power, dtype):
    """ Calculate a sum of distance samples for various dimensions and norms

    Generates a batch of *n_samples* of pairs of points, and uses those points
    to build a table of 1 to *max_dim* rows and 1 to *max_power* columns,
    where each entry is the sum across all samples of the distance metric
    with the given power for pairs of points in the given dimensional
    hypercube.
    """
    # create two lists of random points and their vector difference,
    # where the first dimension is the sample number and the second the
    # coordinates for the sample
    zero = tf.zeros((), dtype=dtype) # used to pass in dtype to Uniform
    x1s = tfd.Uniform(low=zero).sample((n_samples, max_dim))
    x2s = tfd.Uniform(low=zero).sample((n_samples, max_dim))
    vector_difference = x2s - x1s

    # add a third dimension to the tensor which is the vector coordinate
    # raised to ascending powers (eg x, x^2, x^3...)
    sum_terms = tf.abs(tfm.cumprod(tf.tile(
        tf.reshape(vector_difference, (n_samples, max_dim, 1)),
        (1, 1, max_power)), axis=2))

    # generate cumulative sums along the coordinate (second) dimension
    # and raise the sum to the 1/n power (where n is the third dimension)
    # to generate a new tensor where the first dimension is the samples,
    # the second the dimension of the hypercube, and the third the power
    # of the norm.
    norm = tfm.pow(tfm.cumsum(sum_terms, axis=1),
        tf.reshape(1/tf.cast(tf.range(1, max_power + 1), dtype=dtype),
            (1, 1, max_power)))

    # return the sum of the norms
    return tf.reduce_sum(norm, axis=0)

if __name__ == '__main__':
    # parse the arguments
    arguments = docopt(__doc__)
    max_power = int(arguments['-p'])
    max_dim = int(arguments['-d'])
    n_samples = int(arguments['-r'])
    max_batch_size = int(int(arguments['-b']) / max_dim / max_power)
    dtype = tf.float16 if arguments['-f'] == 'half' else (
            tf.float32 if arguments['-f'] == 'single' else (
            tf.float64 if arguments['-f'] == 'double' else (
                None)))
    assert dtype != None, f'Invalid datatype for -f "' + arguments["-f"] + '"'

    # Generate samples in batches that will fit into the GPU's memory,
    # accumulating the sum of distances for each table entry as we go.
    sample_sums = tf.zeros((max_dim, max_power), dtype=dtype)
    sample_count = 0

    while sample_count < n_samples:
        batch_size = min(n_samples - sample_count, max_batch_size)
        sample_sums += sample_batch(batch_size, max_dim, max_power, dtype)
        sample_count += batch_size

    # divide each entry by the number of samples to convert the sums to means
    means = sample_sums / n_samples

    # normalize to the longest diagonal if requested
    if (arguments['-n']):
        means /= tf.pow(
            tf.reshape(tf.cast(tf.range(1, max_dim + 1), dtype=dtype),
                (max_dim, 1)),
            tf.reshape(1/tf.cast(tf.range(1, max_power + 1), dtype=dtype),
                (1, max_power)))

    # display the table in CSV form
    print('DIM\Power, ' + ', '.join([str(i) for i in range(1, max_power + 1)]))
    for i in range(means.shape[0]):
        print(f'{i+1}, ' + ', '.join([f'{x:.6f}' for x in means[i]]))
