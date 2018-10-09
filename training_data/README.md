# Preparing Training Data

## Download the MNIST Handwritten Digit Database

```sh
./download_mnist.sh
```

This will download the MNIST handwritten digit database to the current working
directory.

## Load the Training Data to SharedArray

> Make sure SharedArray package is installed.

Run

```sh
python ./load_mnist_to_sa.py ./mnist/ --merge --binary
```

This will load and binarize the MNIST digits and save them to shared memory via
SharedArray package.
