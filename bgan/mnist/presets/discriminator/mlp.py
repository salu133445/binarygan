"""Network architecture for the discriminator based on MLPs.
"""
NET_D = {}

NET_D['shared'] = [
    ('reshape', 784),                   # 0
    ('dense', 512, 'bn', 'lrelu'),      # 1
    ('dense', 256, 'bn', 'lrelu'),      # 2
    ('dense', 1),                       # 3
]
