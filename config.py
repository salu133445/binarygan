"""Define configuration variables in experiment, model and training levels.

Quick Setup
===========
Change the values in the dictionary `SETUP` for a quick setup.
Documentation is provided right after each key.

Configuration
=============
More configuration options are provided as a dictionary `CONFIG`.
`CONFIG['exp']`, `CONFIG['data']`, `CONFIG['model']`, `CONFIG['train']` and
`CONFIG['tensorflow']` define experiment-, data-, model-, training-,
TensorFlow-related configuration variables, respectively.

Note that the automatically-determined experiment name is based only on the
values defined in the dictionary `SETUP`, so remember to provide the experiment
name manually if you have changed the configuration so that you won't overwrite
existing experiment directories.
"""
import os
import shutil
import distutils.dir_util
import importlib
import tensorflow as tf

# Quick setup
SETUP = {
    'model': 'binarygan',
    # {'binarygan', 'gan'}
    # The model to use. Currently support BinaryGAN and GAN models.

    'exp_name': None,
    # The experiment name. Also the name of the folder that will be created
    # in './exp/' and all the experiment-related files are saved in that
    # folder. None to determine automatically. The automatically-
    # determined experiment name is based only on the values defined in the
    # dictionary `SETUP`, so remember to provide the experiment name manually
    # (so that you won't overwrite a trained model).

    'training_data': 'herman_binarized_mnist_x',
    # Filename of the training data. The training data can be loaded from a npy
    # file in the hard disk or from the shared memory using SharedArray package.

    'training_data_location': 'sa',
    # Location of the training data. 'hd' to load from a npy file stored in the
    # hard disk. 'sa' to load from shared array using SharedArray package.

    'gpu': '0',
    # The GPU index in os.environ['CUDA_VISIBLE_DEVICES'] to use.

    'prefix': 'fix_gan_loss',
    # Prefix for the experiment name. Useful when training with different
    # training data to avoid replacing the previous experiment outputs.

    'sample_along_training': True,
    # True to generate samples along the training process. False for nothing.

    'evaluate_along_training': True,
    # True to run evaluation along the training process. False for nothing.

    'verbose': False,
    # True to print each batch details to stdout. False to print once an epoch.

    'pretrained_dir': None,
    # The directory containing the pretrained model. None to retrain the
    # model from scratch.

    'gan_type': 'gan',
    # {'gan', 'wgan', 'wgan-gp'}
    # The type of GAN objective to use. Currently support GAN, Wasserstein GAN
    # (WGAN), Wasserstein GAN with gradient penalties (WGAN-GP).

    'optimizer': 'adam',
    # {'adam', 'rmsprop'}
    # The optimizer to use. Currently support Adam and RMSProp optimizers.

    'preset_g': 'mlp_bernoulli',
    # BinaryGAN: {'mlp_bernoulli', 'mlp_round', 'cnn_bernoulli', 'cnn_round'}
    # GAN: {'mlp_real', 'cnn_real'}
    # Use a preset network architecture for the generator or set to None and
    # setup `CONFIG['model']['net_g']` to define the network architecture.

    'preset_d': 'mlp',
    # {'mlp', 'cnn', 'mlp_bn', 'cnn_bn'}
    # Use a preset network architecture for the discriminator or set to None
    # and setup `CONFIG['model']['net_d']` to define the network architecture.
}

CONFIG = {}

#===============================================================================
#=========================== TensorFlow Configuration ==========================
#===============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = SETUP['gpu']
CONFIG['tensorflow'] = tf.ConfigProto()
CONFIG['tensorflow'].gpu_options.allow_growth = True

#===============================================================================
#========================== Experiment Configuration ===========================
#===============================================================================
CONFIG['exp'] = {
    'model': None,
    'exp_name': None,
    'pretrained_dir': None,
}

for key in ('model', 'pretrained_dir'):
    if CONFIG['exp'][key] is None:
        CONFIG['exp'][key] = SETUP[key]

# Set default experiment name
if CONFIG['exp']['exp_name'] is None:
    if SETUP['exp_name'] is not None:
        CONFIG['exp']['exp_name'] = SETUP['exp_name']
    else:
        CONFIG['exp']['exp_name'] = '_'.join(
            (SETUP['prefix'], SETUP['model'], SETUP['gan_type'],
             'g', SETUP['preset_g'], 'd', SETUP['preset_d'])
        )

#===============================================================================
#============================= Data Configuration ==============================
#===============================================================================
CONFIG['data'] = {
    'training_data': None,
    'training_data_location': None,
}

for key in ('training_data', 'training_data_location'):
    if CONFIG['data'][key] is None:
        CONFIG['data'][key] = SETUP[key]

#===============================================================================
#=========================== Training Configuration ============================
#===============================================================================
CONFIG['train'] = {
    'sample_along_training': None,
    'evaluate_along_training': None,
    'verbose': None,
    'num_epoch': 20,
    'slope_annealing_rate': 1.1,
}

for key in ('verbose', 'sample_along_training', 'evaluate_along_training'):
    if CONFIG['train'][key] is None:
        CONFIG['train'][key] = SETUP[key]

#===============================================================================
#============================= Model Configuration =============================
#===============================================================================
CONFIG['model'] = {
    # Parameters
    'batch_size': 64, # Note: tf.layers.conv3d_transpose requires a fixed batch
                      # size in TensorFlow < 1.6
    'gan': {
        'type': None, # 'gan', 'wgan', 'wgan-gp'
        'clip_value': .01,
        'gp_coefficient': 10.
    },
    'optimizer': {
        'type': None,
        'lr': .0001,
        'epsilon': 1e-8,
        # Parameters for Adam optimizers
        'beta1': .5,
        'beta2': .9,
        # Parameters for RMSProp optimizers
        'momentum': 0.0,
        'decay': .9,
    },

    # Data
    'out_width': 28,
    'out_height': 28,
    'out_channel': 1,

    # Network architectures (define them here if not using the presets)
    'net_g': None,
    'net_d': None,
    'net_r': None,

    # Samples
    'num_sample': 64,
    'sample_grid': (8, 8),

    # Directories
    'checkpoint_dir': None,
    'sample_dir': None,
    'eval_dir': None,
    'log_dir': None,
    'src_dir': None,
}

if CONFIG['model']['gan']['type'] is None:
    CONFIG['model']['gan']['type'] = SETUP['gan_type']
if CONFIG['model']['optimizer']['type'] is None:
    CONFIG['model']['optimizer']['type'] = SETUP['optimizer']

# Import preset network architectures
if CONFIG['model']['net_g'] is None:
    IMPORTED = importlib.import_module('.'.join((
        'bgan.mnist.presets', 'generator', SETUP['preset_g']
    )))
    CONFIG['model']['net_g'] = IMPORTED.NET_G
if CONFIG['model']['net_d'] is None:
    IMPORTED = importlib.import_module('.'.join((
        'bgan.mnist.presets', 'discriminator', SETUP['preset_d']
    )))
    CONFIG['model']['net_d'] = IMPORTED.NET_D

# Set default directories
for kv_pair in (('checkpoint_dir', 'checkpoints'), ('sample_dir', 'samples'),
                ('eval_dir', 'eval'), ('log_dir', 'logs'), ('src_dir', 'src')):
    if CONFIG['model'][kv_pair[0]] is None:
        CONFIG['model'][kv_pair[0]] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'exp', SETUP['model'],
            CONFIG['exp']['exp_name'], kv_pair[1])

#===============================================================================
#=================== Make directories & Backup source code =====================
#===============================================================================
# Make sure directories exist
for path in (CONFIG['model']['checkpoint_dir'], CONFIG['model']['sample_dir'],
             CONFIG['model']['eval_dir'], CONFIG['model']['log_dir'],
             CONFIG['model']['src_dir']):
    if not os.path.exists(path):
        os.makedirs(path)

# Backup source code
for path in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if os.path.isfile(path):
        if path.endswith('.py'):
            shutil.copyfile(os.path.basename(path),
                            os.path.join(CONFIG['model']['src_dir'],
                                         os.path.basename(path)))

distutils.dir_util.copy_tree(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bgan'),
    os.path.join(CONFIG['model']['src_dir'], 'bgan')
)
