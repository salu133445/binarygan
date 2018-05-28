"""Network architecture for the generator based on MLPs.
"""
NET_G = {}

NET_G['z_dim'] = 128

NET_G['shared'] = [
    ('dense', (256), 'bn', 'lrelu'),        # 0
    ('dense', (512), 'bn', 'lrelu'),        # 1
    ('dense', (1024), 'bn', 'lrelu'),       # 2
    ('dense', (784), 'bn', 'sigmoid'),      # 3
    ('reshape', (28, 28, 1)),               # 4
]
