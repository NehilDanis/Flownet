from torch.utils.data import Dataset
from imageio import imread
from typing import Callable, Optional
import numpy as np


def flying_chairs_loader(sample):
    """
    will create a numpy array of two input images and one target flow data
    :param sample: one sample from the flying chairs dataset, has the format [[img1_path, img2_path], flow_path]
    :return:
    """
    inputs, target = sample[0], sample[1]
    img1, img2 = imread(inputs[0]).astype(np.float32), imread(inputs[1]).astype(np.float32)

    with open(target, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return [img1, img2], data2D


class CustomDataset(Dataset):
    def __init__(self, file_names: [str],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader=flying_chairs_loader
                 ):
        """
        Initialized the custom datasets
        :param file_names: list of absolute paths to samples along with path to their targets
        :param transform: transformation for inputs
        :param target_transform: transformation for target
        :param loader: a helper function to read the inputs and targets into numpy array
        """
        self.file_names = file_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, idx):
        """
        will return the sample with index idx from the dataset
        :param idx: the index of the sample which will be returned
        :return: input, target pair
        """
        inputs, target = self.loader(self.file_names[idx])
        # apply the transforms and target transform in case they exist
        if self.transform:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        """
        will return the total number of samples in the dataset.
        :return: number of samples in the dataset
        """
        return len(self.file_names)