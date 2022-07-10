"""The file is for collecting and extracting MNIST dataset images for training and testing."""
import os
import sys
import functools
import operator
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve


# from: https://github.com/datapythonista/mnist
DATASET_DIRECTORY = "../data/"
URL = "http://yann.lecun.com/exdb/mnist/"


def parse_idx(fd):
    DATA_TYPES = {
        0x08: "B",  # unsigned byte
        0x09: "b",  # signed byte
        0x0B: "h",  # short (2 bytes)
        0x0C: "i",  # int (4 bytes)
        0x0D: "f",  # float (4 bytes)
        0x0E: "d",
    }  # double (8 bytes)
    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError(
            "Invalid IDX file, file empty or does not contain a full header."
        )

    zeros, data_type, num_dimensions = struct.unpack(">HBB", header)
    if zeros != 0:
        raise IdxDecodeError(
            "Invalid IDX file, file must start with two zero bytes. "
            "Found 0x%02x" % zeros
        )

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError("Unknown data type 0x%02x in IDX file" % data_type)

    dimension_sizes = struct.unpack(
        ">" + "I" * num_dimensions, fd.read(4 * num_dimensions)
    )
    data = array.array(data_type, fd.read())
    data.byteswap()
    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError(
            "IDX file has wrong number of items. "
            "Expected: %d. Found: %d" % (expected_items, len(data))
        )

    return np.array(data).reshape(dimension_sizes)


def print_download_progress(count, block_size, total_size):
    pct_complete = int(count * block_size * 100 / total_size)
    pct_complete = min(pct_complete, 100)
    msg = "\r- Download progress: %d" % (pct_complete) + "%"
    sys.stdout.write(msg)
    sys.stdout.flush()


def download_and_parse_mnist_file(fname, target_dir=None, force=False):
    if not os.path.exists(DATASET_DIRECTORY):
        os.makedirs(DATASET_DIRECTORY)
    if not os.path.exists(DATASET_DIRECTORY + fname):
        print("Downloading " + fname)
        file_path = os.path.join(DATASET_DIRECTORY, fname)
        url = URL + fname
        file_path, _ = urlretrieve(
            url=url, filename=file_path, reporthook=print_download_progress
        )
        print("\nDownload finished.")

    fopen = gzip.open if os.path.splitext(fname)[1] == ".gz" else open
    with fopen(fname, "rb") as fd:
        return parse_idx(fd)


def train_images():
    return download_and_parse_mnist_file("../data/train-images-idx3-ubyte.gz")


def test_images():
    return download_and_parse_mnist_file("../data/t10k-images-idx3-ubyte.gz")


def train_labels():
    return download_and_parse_mnist_file("../data/train-labels-idx1-ubyte.gz")


def test_labels():
    return download_and_parse_mnist_file("../data/t10k-labels-idx1-ubyte.gz")
