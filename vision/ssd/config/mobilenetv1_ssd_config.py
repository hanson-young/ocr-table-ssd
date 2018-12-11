import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 768
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# specs = [
#     SSDSpec(19, 16, SSDBoxSizes(60, 120), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
# ]

specs = [
    SSDSpec(48, 16, SSDBoxSizes(60, 120), [2, 3]),

    SSDSpec(24, 32, SSDBoxSizes(120, 240), [2, 3]),
    SSDSpec(12, 64, SSDBoxSizes(240, 360), [2, 3]),
    SSDSpec(6, 128, SSDBoxSizes(360, 480), [2, 3]),
    SSDSpec(3, 256, SSDBoxSizes(480, 640), [2, 3]),
    SSDSpec(2, 384, SSDBoxSizes(640, 800), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)
print(priors)