import random
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import h5py


class Fives(Dataset):
    def __init__(self, file_path, t: str, transform=None):
        hf = h5py.File(file_path, "r")

        self.transform = transform

        if t in ["train", "val", "test", "cal"]:
            self.images = hf[t]["images"]
            self.segmentations = hf[t]["label"]
            self.id = hf[t]["id"]
            if t + "/ood" in hf:
                self.ood = hf[t]["ood"]
            if t + "/quality" in hf:
                self.quality = hf[t]["quality"]
        else:
            raise ValueError(f"Unknown test/train/val/cal specifier: {t}")

    def __getitem__(self, index):

        image = self.images[index]
        segmentation = self.segmentations[index]

        if self.transform == None:
            pass
        else:
            if random.random() > 0.5:
                image, segmentation = self.transform(image, segmentation)

        # normalise image
        image = (image - image.mean(axis=(0, 1))) / image.std(axis=(0, 1))

        # change shape from (size, size, 3) to (3, size, size)
        image = np.moveaxis(image, -1, 0)

        # Convert to torch tensor
        image = torch.from_numpy(image)
        segmentation = torch.from_numpy(segmentation)

        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        segmentation = segmentation.type(torch.LongTensor)

        return image, segmentation

    def __len__(self):
        return self.images.shape[0]

    def get_id(self, i):
        return self.id[i]
