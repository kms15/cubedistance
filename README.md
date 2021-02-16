# cubedistance

Find the average distance between two random points in an n-dimensional cube
using a Monte Carlo approach on a GPU.

This was an exercise in parallel computation using TensorFlow inspired by Paul
Alfille's [more in-depth work](https://github.com/alfille/distance) which
includes a CPU based Monte Carlo calculation.  The python script in this
repository is designed to accept the same parameters and generate similar
output as the C code in the linked project, but generally runs much faster when
run with larger samples sizes (I've seen as much as a 400x speedup using a fast
GPU).

Three significant trade offs are made to achieve this:

 1. Calculations are done in 32-bit precision (as most GPUs do not support
    64-bit precision).

 2. More dependencies are required (e.g. TensorFlow, TensorFlow-Probability,
    and potentially libraries like libcudnn for GPU support).

 3. The code is less intuitive as it involves manipulating 3 dimensional
    tensors (but this is precisely what allows it to run well on massively
    parallel architectures such as a GPUs).

## Installation

First, download a local copy of the project

    git clone http://github/kms15/cubedistance
    cd cubedistance

This python script requires TensorFlow, TensorFlow-Probability, and docopt.
If you do not already have these in your environment, you may want to create
a and activate virtual environment in which to install these dependencies:

    python3 -m venv myenv
    source/myenv/bin/activate

You can then install these dependencies using pip3:

    pip3 install --upgrade pip3
    pip3 install tensorflow tensorflow-probability docopt

## GPU support

GPU support in TensorFlow continues to be a bit of a hassle.  I will refer you
to the
[official TensorFlow documentation](https://www.tensorflow.org/install/gpu)
for more details.

## Usage

    python3 cubedistance.py -h

will show the following usage instructions:

    cubedistance.py - find the average distance between random points in a
        hypercube using a Monte Carlo approach.

    By Kendrick Shaw 2021 -- MIT License

    Usage:
      cubedistance.py [-d <max_dim>] [-p <max_power>] [-r <num_samples>]
          [-b <batch_size>] [-n]
      cubedistance.py (-h | --help)

    Options:
      -h --help         Show this screen
      -d <max_dim>      Maximum hypercube dimensions [default: 100]
      -p <max_power>    Maximum power of the p-norm [default: 3]
      -r <num_samples>  Number of Monte Carlo samples for each entry
                        [default: 100000000]
      -b <batch_size>   The maximum number of parallel values computed in each
                        batch, equal to the number of samples in the batch times
                        max_dim and max_power. [default: 100000000]
      -n                Normalize to the longest diagonal

## Examples

The following will run the program with its default parameters (i.e. one million100
dimensions with up to the L3-norm):

    python3 cubedistance.py

A more complex example equivalent to the one from Dr. Alfille's project:

    python3 cubedistance.py -p 10 -r 100000 -d 200 -n > Sample.csv
