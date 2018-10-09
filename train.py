"""Script for training the model."""
import os
import numpy as np
import tensorflow as tf
from bgan.mnist.models import BinaryGAN, GAN
from config import CONFIG

def load_data():
    """Load and return the training data."""
    print('[*] Loading data...')

    # Load data from SharedArray
    if CONFIG['data']['training_data_location'] == 'sa':
        import SharedArray as sa
        x_train = sa.attach(CONFIG['data']['training_data'])

    # Load data from hard disk
    elif CONFIG['data']['training_data_location'] == 'hd':
        if os.path.isabs(CONFIG['data']['training_data']):
            x_train = np.load(CONFIG['data']['training_data'])
        else:
            filepath = os.path.abspath(os.path.join(
                os.path.realpath(__file__), 'training_data',
                CONFIG['data']['training_data']))
            x_train = np.load(filepath)

    return x_train

def main():
    """Main function."""
    if CONFIG['exp']['model'] not in ('binarygan', 'gan'):
        raise ValueError("Unrecognizable model name")

    print("Start experiment: {}".format(CONFIG['exp']['exp_name']))

    # Load training data
    x_train = load_data()

    # Open TensorFlow session
    with tf.Session(config=CONFIG['tensorflow']) as sess:

        # Create model
        if CONFIG['exp']['model'] == 'gan':
            gan = GAN(sess, CONFIG['model'])
        elif CONFIG['exp']['model'] == 'binarygan':
            gan = BinaryGAN(sess, CONFIG['model'])

        # Initialize all variables
        gan.init_all()

        # Load pretrained model if given
        if CONFIG['exp']['pretrained_dir'] is not None:
            gan.load_latest(CONFIG['exp']['pretrained_dir'])

        # Train the model
        gan.train(x_train, CONFIG['train'])

if __name__ == '__main__':
    main()
