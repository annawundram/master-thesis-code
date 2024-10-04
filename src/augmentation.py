import random
from torchvision.transforms import v2
import torch
from PIL import Image
import numpy as np

def apply_augmentation(image, segmentation):
    # apply random augmentation
    # return transformed image and segmentation
    type = random.randint(0,8)
    angle = random.randrange(0, 360)

    image = Image.fromarray(image)
    segmentation = Image.fromarray(segmentation)

    if type == 0:
        transform = v2.ColorJitter(brightness=(0.5, 1.5))
        transformed = transform(image)
    elif type == 1:
        transform = v2.ColorJitter(hue=(0, 0.1))
        transformed = transform(image)
    elif type == 2:
        transform = v2.ColorJitter(saturation=(0.3, 1.5))
        transformed = transform(image)
    elif type == 3:
        transform = v2.ColorJitter(contrast=(0.4, 2))
        transformed = transform(image)
    elif type == 4:
        transformed = v2.functional.horizontal_flip(image)
        segmentation = v2.functional.horizontal_flip(segmentation)
    elif type == 5:
        transformed = v2.functional.vertical_flip(image)
        segmentation = v2.functional.vertical_flip(segmentation)
    elif type == 6:
        transformed = v2.functional.rotate(image,angle)
        segmentation = v2.functional.rotate(segmentation, angle)
    else:
        transform = v2.GaussianBlur(kernel_size=(3, 3), sigma=(1., 9.))
        transformed = transform(image)

    image = np.asarray(transformed).copy()
    segmentation = np.asarray(segmentation).copy()

    return image, segmentation

def MixUp(x,y, alpha = 1):
    # get lam
    # permute batch dimension
    # subtract permuted x/y from x/y
    batch_size = x.size()[0]
    idx = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)

    x = lam * x + (1 - lam) * x[idx]
    y = lam * y + (1 - lam) * y[idx]

    return x, y

