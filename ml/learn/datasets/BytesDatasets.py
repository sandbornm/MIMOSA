from os.path import join
import numpy as np
import torch
import glob
import pandas as pd
from torch.utils.data import Dataset


class MalwareBytesDataset(Dataset):
    def __init__(self, examples_dir: str, labels_csv: str, sz=2**10):
        """
        Dataset class malware binaries for deep learning multilabel classification
        """
        self.examples_dir = examples_dir
        self.labels_csv = labels_csv
        self.sz = sz

        self.examples = glob.glob(join(examples_dir, '*.bin'))
        self.labels = pd.read_csv(labels_csv) if '.tsv' not in labels_csv else pd.read_csv(labels_csv, sep='\t')
        self.hashes = self.labels['sha1sum'].tolist()

        self.classes = self.labels.columns[1:]
        self.n_examples = len(self.labels)
        self.n_classes = len(self.labels.columns) - 1

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hash = self.hashes[idx]

        # get hash's bin and read bytes into array
        path = join(self.examples_dir, hash+'.bin')
        dtype = np.dtype('B')
        with open(path, "rb") as f:
            bytes = np.fromfile(f, dtype)

        if self.sz <= len(bytes):
            bytes = bytes[:self.sz]
        else:
            bytes = np.hstack([bytes, np.zeros(self.sz - len(bytes), dtype)])

        bytes = torch.from_numpy(bytes.astype(float)).float()

        # get the label
        label = self.labels.loc[self.labels['sha1sum'] == hash]
        label = label[self.classes].to_numpy()

        return {'hash': hash, 'example': bytes, 'label': label}
