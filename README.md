# cubedistance

Find the average distance between two random points in an n-dimensional cube
using a Monte Carlo approach on a GPU.

This was an exercise in parallel computation using TensorFlow inspired by Paul
Alfille's [more in-depth work](https://github.com/alfille/distance) which
includes a CPU based Monte Carlo calculation.  The python script in this
repository is designed to accept the same parameters and generate similar
output as the C code in the linked project, but generally runs much faster when
run with larger samples sizes.

Two significant trade offs are made to achieve this:

 1. More dependencies are required (e.g. TensorFlow, TensorFlow-Probability,
    and potentially libraries like libcudnn for GPU support).

 2. The code is less intuitive as it involves manipulating 3 dimensional
    tensors (but this is precisely what allows it to run well on massively
    parallel architectures such as a GPUs).

## Informal Benchmarks

We first consider benchmarks is from a P1 laptop with an Intel i9-9880H CPU,
Nvidia T2000 GPU, and 64 GB of RAM:

Device | Command                                        | Time (s) | Speedup
:-----:|:-----------------------------------------------|---------:|-------:
  CPU  | ./distance -r 100000000                        | 8143.240 |  1.0
  CPU  | python3 cubedistance.py -r 100000000 -f double |  318.195 | 25.6
  CPU  | python3 cubedistance.py -r 100000000 -f single |  239.116 | 34.1
  GPU  | python3 cubedistance.py -r 100000000 -f double |  253.708 | 32.1
  GPU  | python3 cubedistance.py -r 100000000 -f single |   96.461 | 84.0

These show a fairly significant speedup, reflecting a combination of greater
reuse of intermediate results and taking advantage of the parallel
computational hardware available in the CPU and GPU. Despite the limited
support for double precision in the GPU, the computation still runs a bit
faster on the GPU even in double precision.  The performance difference between
single and double precision on the GPU is less than I would have expected,
suggesting that memory bandwidth may be a limiting factor.

We next consider benchmarks for a larger server, with dual Epyc 7742
processors, multiple Nvidia 2080 Ti's (only one is used for this benchmark),
and 1024 GB of RAM:

Device | Command                                                      | Time (s) | Speedup
:-----:|:-------------------------------------------------------------|---------:|-------:
  CPU  | ./distance -r 100000000                                      | 9047.369 |   1.0
  CPU  | python3 cubedistance.py -r 100000000 -f double               |  244.071 |  37.1
  CPU  | python3 cubedistance.py -r 100000000 -f single               |  237.488 |  38.1
  CPU  | python3 cubedistance.py -r 100000000 -f double -b 2000000000 |   42.957 | 210.6
  CPU  | python3 cubedistance.py -r 100000000 -f single -b 2000000000 |   25.718 | 351.8
  GPU  | python3 cubedistance.py -r 100000000 -f double               |   41.590 | 217.5
  GPU  | python3 cubedistance.py -r 100000000 -f single               |   15.293 | 591.6

These results show a larger speedup, reflecting the larger number of parallel
resources available on this hardware.  Note that the single threaded result is
actually slower than we saw previously on the laptop, reflecting a lower boost
clock speed on the server CPU.  The CPU results with the default batch size
appear to be limited by resource contention between the many cores; setting the
batch size to a larger value (using -b) appears to relieve this contention
leading to better performance. (Increasing the batch size did not improve
performance on the laptop, results not shown).  Note that the single
consumer-grade GPU is able to outperform the much more expensive server CPUs,
even at double precision.  Further performance increases should be possible by
taking advantage of the multiple GPUs in this machine.

The fine print: These tests were run with a minimal amount of rigor, and thus
the results should be viewed as informal; the reader should run benchmarks
on their own hardware before making critical decisions based on them. The tests
were run with distance commit 9bea773ac859fef193297b60a5b9cb06737ba181 compiled
with gcc -Ofast and cubedistance commit
44555032aac32b2d0cd108226605b2778a43f70c.  Timing was done with the unix `time`
command and only a single run was done of each benchmark.  No attempts were
made to control for other processes on the machines (although the machines were
largely otherwise idle), and the exact versions of all libraries used was not
recorded.  GPU vs CPU was controlled by running the command within or outside
of a container with the required NVidia CUDA libraries.

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

## Examples

The following will run the program with its default parameters (i.e. one
million samples, up to one hundred hypercube dimensions with up to the
L3-norm):

    python3 cubedistance.py

A more complex example equivalent to the one from Dr. Alfille's project:

    python3 cubedistance.py -p 10 -r 100000 -d 200 -n > Sample.csv
