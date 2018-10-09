"""Network architecture for the discriminator for the real-valued model
implemented by CNNs."""
NET_D = {}

NET_D['main'] = [
    ('conv2d', (32, (3, 3), (1, 1)), 'bn', 'lrelu'), # 0 (26, 26)
    ('maxpool2d', ((2, 2), (2, 2))),                 # 1 (13, 13)
    ('conv2d', (64, (3, 3), (1, 1)), 'bn', 'lrelu'), # 2 (11, 11)
    ('maxpool2d', ((2, 2), (2, 2), 'same')),         # 3 (6, 6)
    ('reshape', (64*6*6)),                           # 4
    ('dense', (128), 'bn', 'lrelu'),                 # 5
    ('dense', 1),                                    # 6
]
