import numpy as np
from itertools import product
from functions import clippingForNumpy

class PatchGenerator:
    def __init__(self, array, patch_size, slide):
        assert array.ndim == len(patch_size) == len(slide)

        self.array = array
        self.patch_size = patch_size

        max_size = np.array(self.array.shape) - self.patch_size + 1
        ranges = []
        for i in range(self.array.ndim):
            r = range(0, max_size[i], slide[i])
            ranges.append(r)

        self.indices = [i for i in product(*ranges)]

    def __len__(self):
        return len(self.indices)

    def __call__(self):
        for index in self.indices:
            lower_clip_size = np.array(index)
            upper_clip_size = lower_clip_size + self.patch_size

            patch_array = clippingForNumpy(self.array, lower_clip_size, upper_clip_size)

            yield index, patch_array


def isMasked(array):
    if (array != 0).any():
        return True
    else:
        return False
