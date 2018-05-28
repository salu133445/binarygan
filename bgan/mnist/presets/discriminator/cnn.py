"""Network architecture for the discriminator based on CNNs.
"""
NET_D = {}

NET_D['shared'] = [
    ('dense', (128*7*7), 'bn', 'relu'),                     # 0
    ('transconv2d', (64, (3, 3), (2, 2)), 'bn', 'relu')     # 1
    ('transconv2d', (32, (3, 3), (2, 2)), 'bn', 'relu')     # 1
    ('transconv2d', (64, (3, 3), (2, 2)), 'bn', 'relu')     # 1
    ('transconv2d', (64, (3, 3), (2, 2)), 'bn', 'relu')     # 1
    ('dense', 1, 'bn', 'relu'),                             # 3
]
