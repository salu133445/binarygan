"""Load and save the MNIST dataset to shared memory via SharedArray package."""
import os
import argparse
import numpy as np
import SharedArray as sa

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', help="Root directory of the dataset.")
    parser.add_argument('--prefix', default='',
                        help="Prefix to the file name to save in SharedArray.")
    parser.add_argument('--merge', help='Merge train and test set',
                        action='store_true')
    parser.add_argument('--binary', help='Binarize the data.',
                        action='store_true')
    parser.add_argument('--labels', help='Store the labels as well.',
                        action='store_true')
    parser.add_argument('--onehot', help='Use onehot encoding for the labels',
                        action='store_true')
    args = parser.parse_args()
    return (args.dataset_root, args.prefix, args.merge, args.binary,
            args.labels, args.onehot)

def save_to_sa(name, data):
    """Save data to SharedArray."""
    arr = sa.create(name, data.shape, data.dtype)
    np.copyto(arr, data)

def load(dataset_root, prefix, merge, binary, labels, onehot):
    """Load and save the dataset to SharedArray."""
    with open(os.path.join(dataset_root, 'train-images-idx3-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1))
        if binary:
            trX = (trX > 0)
            if not merge:
                save_to_sa('_'.join((prefix, 'binarized_mnist_x_train')), trX)
        elif not merge:
            save_to_sa('_'.join((prefix, 'mnist_x_train')), trX)

    with open(os.path.join(dataset_root, 't10k-images-idx3-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1))
        if binary:
            teX = (teX > 0)
            if not merge:
                save_to_sa('_'.join((prefix, 'binarized_mnist_x_test')), teX)
        elif not merge:
            save_to_sa('_'.join((prefix, 'mnist_x_test')), teX)

    if merge:
        if binary:
            filename = '_'.join((prefix, 'binarized_mnist_x'))
        else:
            filename = '_'.join((prefix, 'mnist_x'))
        save_to_sa(filename, np.concatenate((trX, teX)))

    if not labels:
        return

    with open(os.path.join(dataset_root, 'train-labels-idx1-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        trY = loaded[8:].reshape((60000))
        if onehot:
            onehot_encoded = np.zeros((60000, 10), np.bool_)
            onehot_encoded[np.arange(60000), trY] = True
            trY = onehot_encoded
            if not merge:
                save_to_sa('_'.join((prefix, 'mnist_y_train_onehot')), trY)
        elif not merge:
            save_to_sa('_'.join((prefix, 'mnist_y_train')), trY)

    with open(os.path.join(dataset_root, 't10k-labels-idx1-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        teY = loaded[8:].reshape((10000))
        if onehot:
            onehot_encoded = np.zeros((10000, 10), np.bool_)
            onehot_encoded[np.arange(10000), teY] = True
            teY = onehot_encoded
            if not merge:
                save_to_sa('_'.join((prefix, 'mnist_y_test_onehot')), teY)
        elif not merge:
            save_to_sa('_'.join((prefix, 'mnist_y_test')), teY)

    if merge:
        if onehot:
            filename = '_'.join((prefix, 'mnist_y_onehot'))
        else:
            filename = '_'.join((prefix, 'mnist_y'))
        save_to_sa(filename, np.concatenate((trY, teY)))

def main():
    """Main function"""
    dataset_root, prefix, merge, binary, labels, onehot = parse_arguments()
    load(dataset_root, prefix, merge, binary, labels, onehot)

if __name__ == '__main__':
    main()
