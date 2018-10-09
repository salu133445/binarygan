"""Network architecture for the generator for the proposed model implemented by
MLPs and DBNs.
"""
NET_G = {}

NET_G['z_dim'] = 128

NET_G['main'] = [
    ('dense', (1024), 'bn', 'relu'),  # 0
    ('dense', (784), 'bn', 'round'),  # 1
    ('reshape', (28, 28, 1)),         # 2
]
