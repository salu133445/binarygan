# BinaryGAN

## Prepare Training Data

- Download MNIST database by running the script:

  ```sh
  ./training_data/download_mnist.sh
  ```

- or download it manually:
  1. Download MNIST database [here](http://yann.lecun.com/exdb/mnist/)
  2. Decompress all the `.gz` files
  3. Move the decompressed files to `./training_data/mnist`

- Store the data to shared memory (optional)

  > Make sure the SharedArray package has been installed.

  ```sh
  python ./training_data/load_mnist_to_sa.py ./training_data/mnist/ \
  --merge --binary
  ```

## Configuration

Modify `config.py` for configuration.

- Quick setup

  Change the values in the dictionary `SETUP` for a quick setup. Documentation
  is provided right after each key.

- More configuration options

  Four dictionaries `EXP_CONFIG`, `DATA_CONFIG`, `MODEL_CONFIG` and
  `TRAIN_CONFIG` define experiment-, data-, model- and training-related
  configuration variables, respectively.

  > The automatically-determined experiment name is based only on the values
defined in the dictionary `SETUP`, so remember to provide the experiment name
manually when you modify any other configuration variables so that you won't
overwrite a trained model.

## Train the model

```sh
python train.py
```
