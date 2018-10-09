"""Network architecture for the generator for the proposed model implemented by
CNNs and DBNs.
"""
NET_G = {}

NET_G['z_dim'] = 128

NET_G['main'] = [
    ('reshape', (1, 1, 128)),                             # 0 (1, 1)
    ('transconv2d', (128, (2, 2), (1, 1)), 'bn', 'relu'), # 1 (2, 2)
    ('transconv2d', (64, (4, 4), (2, 2)), 'bn', 'relu'),  # 2 (6, 6)
    ('transconv2d', (32, (3, 3), (2, 2)), 'bn', 'relu'),  # 3 (13, 13)
    ('transconv2d', (1, (4, 4), (2, 2)), 'bn', 'round'),  # 4 (28, 28)
]
