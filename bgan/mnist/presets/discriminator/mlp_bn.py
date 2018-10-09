"""Network architecture for the discriminator for the real-valued model
implemented by MLPs."""
NET_D = {}

NET_D['main'] = [
    ('reshape', 784),              # 0
    ('dense', 512, 'bn', 'lrelu'), # 1
    ('dense', 256, 'bn', 'lrelu'), # 2
    ('dense', 1),                  # 3
]
