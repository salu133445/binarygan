# Data

## Binarized MNIST Database

We use the binarized version of the MNIST handwritten digit database.
Specifically, we

- convert pixels with nonzero intensities to ones
- convert pixels with zero intensities to zeros.

The following figure shows some sample binarized MNIST digits seen in our
training data.

<img src="figs/train.png" alt="training_data" style="width:100%; max-width:600px; display:block;">
<p class="caption">Sample binarized MNIST digits</p>
