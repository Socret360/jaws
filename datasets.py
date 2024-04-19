import re
import os
from glob import glob
from torch_geometric.data import Dataset
#
from utils import text_to_graph, preprocess_khpos_sample, preprocess_phylypo_sample

def build_dataset(path: str, type: str) -> Dataset:
    """ Construct the appropriate dataset object from `path`  and `type`.

    Args:
    ---
        - `path`: str
            The path to the dataset. (Text file if `type` is khpos and  a directory if of type `phylypo`)
        - `type`: str
            The type of dataset. (One of type `khpos` and `phylypo`)

    e
    """
    dataset = {
        "khpos": KhPOSDataset,
        "phylypo": PhylypoDataset,
    }[type]
    return dataset(path)


class KhPOSDataset(Dataset):
    """ Pytorch dataset generator for processing KhPOS dataset. The dataset can be found here:
    https://github.com/ye-kyaw-thu/khPOS. The samples in the dataset are listed as single lines
    in a text file.
    """

    def __init__(self, filepath: str) -> None:
        """ Initialize the KhPOS dataset generator.

        Args:
        ---
            - `filepath`: str
                Path pointing to the .txt file containing a list of sample texts.
        """
        with open(filepath, "r") as file:
            self.samples = file.readlines()

    def __len__(self):
        """ Returns the length of the samples.

        Returns:
        ---
            An integer representing the number of samples of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """ Returns a graph representation of the sample with index `idx`.

        Args:
        ---
            `idx`: int - An integer representing the index of the samples.

        Returns:
        ---
            A preprocessed sample as a graph in the form of pytorch `Data` object.
        """
        text = self.samples[idx]
        preprocessed, original = preprocess_khpos_sample(text.strip())
        graph = text_to_graph(preprocessed, original)
        return graph


class PhylypoDataset(Dataset):
    """ Pytorch dataset generator for preprocessing dataset scraped by Phylypo. The dataset can be found here:
    https://github.com/phylypo/segmentation-crf-khmer. The samples in the dataset are listed as text files in a directory.
    This generator assumes that each file contains Khmer text separated by space and each file name is post fix by _seg (e.g. sample1_seg.txt).
    """

    def __init__(self, data_dir: str) -> None:
        """ Initialize Phylypo dataset generator.

        Args:
        ---
            - `data_dir`: str
                Path to the dataset samples folder.
        """
        self.samples = sorted(list(glob(os.path.join(data_dir, "*_seg.txt"))))

    def __len__(self):
        """ Returns the length of the samples.

        Returns:
        ---
            An integer representing the number of samples of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """ Returns a graph representation of the sample with index `idx`.

        Args:
        ---
            `idx`: int
                An integer representing the index of the samples.

        Returns:
        ---
            A preprocessed sample as a graph in the form of pytorch `Data` object.
        """
        with open(self.samples[idx], 'r', encoding='utf-8') as sample_file:
            preprocessed, original = preprocess_phylypo_sample(
                sample_file.read())
            graph = text_to_graph(preprocessed, original)
            return graph
