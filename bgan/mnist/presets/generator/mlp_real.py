"""Network architecture for the generator for the real-valued model implemented
by MLPs."""
NET_G = {}

NET_G['z_dim'] = 128

NET_G['main'] = [
    ('dense', (1024), 'bn', 'relu'),   # 0
    ('dense', (784), 'bn', 'sigmoid'), # 1
    ('reshape', (28, 28, 1)),          # 2
]
