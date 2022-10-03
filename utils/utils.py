import os
import torch
import torch.utils.data as data


def make_dataset_split(dataset: [], split: float = 0.8, random_seed: int = 42):

    """
    Given dataset, creates train and validation sets.
    :param dataset: array of input and target pairs
    :param split: the percentage of training fraction
    :param random_seed: the random state value used in train test split
    :return: train and test/validation data
    """
    num_train_samples = int(len(dataset)*split)
    num_val_samples = len(dataset) - num_train_samples

    seed = torch.Generator().manual_seed(random_seed)
    return data.random_split(dataset, [num_train_samples, num_val_samples], generator=seed)


def get_flying_chairs_data_paths(root: str):

    """
    Collects the path of every input and target of the flying chairs dataset
    :param root: the absolute path to where dataset is located
    :return: return samples in the following format [[<path to img1>, <path to img2>], <path to flow>]
    """

    ''' 
    Format of flying chairs dataset <sample_id>_img1.ppm <sample_id>_img2.ppm <sample_id>_flow.flo
    Triplets should be found
    '''

    samples = []
    for name in sorted(os.listdir(root)):
        if name.endswith('_flow.flo'):
            sample_id = name[: -9]
            img1 = os.path.join(root, sample_id + "_img1.ppm")
            img2 = os.path.join(root, sample_id + "_img2.ppm")
            flow = os.path.join(root, name)
            samples.append([[img1, img2], flow])

    return samples

