"""Network architecture for the generator based on CNNs.
"""
NET_G = {}

NET_G['z_dim'] = 128

NET_G['shared'] = [
    ('dense', (3*512), 'bn', 'relu'),                           # 0
    ('reshape', (3, 1, 1, 512)),                                # 1 (3, 1, 1)
    ('transconv3d', (256, (2, 1, 1), (1, 1, 1)), 'bn', 'relu'), # 2 (4, 1, 1)
    ('transconv3d', (128, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'), # 3 (4, 4, 1)
    ('transconv3d', (128, (1, 1, 3), (1, 1, 3)), 'bn', 'relu'), # 4 (4, 4, 3)
    ('transconv3d', (64, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'),  # 5 (4, 16, 3)
    ('transconv3d', (64, (1, 1, 3), (1, 1, 2)), 'bn', 'relu'),  # 6 (4, 16, 7)
]
