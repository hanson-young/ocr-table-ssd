from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=255.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None, mask = None: (img / std, boxes, labels, mask / std),
            ToTensor(),
        ])


    def __call__(self, img, boxes, labels, masks):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels, masks)


class TestTransform:
    def __init__(self, size, mean=0.0, std=255.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None, mask = None: (img / std, boxes, labels, mask / std),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels, mask):
        return self.transform(image, boxes, labels, mask)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=255.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None, mask = None: (img / std, boxes, labels, mask / std),
            ToTensor()
        ])

    def __call__(self, image, boxes=None, labels=None, mask = None):
        image, _, _, mask = self.transform(image, boxes, labels, mask)
        return image, mask