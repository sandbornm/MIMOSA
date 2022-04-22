from os.path import join
import torch
import cv2
import glob
import pandas as pd
from torch.utils.data import Dataset


class MalwareImageDataset(Dataset):
    def __init__(self, examples_dir: str, labels_csv: str, transform=None, sz=(64,64)):
        """
        Dataset class for images of malware binaries for deep learning multilabel classification
        """
        self.examples_dir = examples_dir
        self.labels_csv = labels_csv
        self.transform = transform
        self.sz = sz

        self.examples = glob.glob(join(examples_dir, '*.png'))
        self.labels = pd.read_csv(labels_csv) if '.tsv' not in labels_csv else pd.read_csv(labels_csv, sep='\t')
        self.hashes = self.labels['sha1sum'].tolist()

        self.configs = self.labels.columns[1:]
        self.n_examples = len(self.labels)
        self.n_classes = len(self.labels.columns) - 1

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hash = self.hashes[idx]

        # get hash's image
        path = join(self.examples_dir, hash+'.png')
        image = cv2.imread(path)
        h, w, c = image.shape
        nh, nw = self.sz
        if nh*nw > h*w:
            image = cv2.resize(image, self.sz, interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, self.sz)

        # get the label
        label = self.labels.loc[self.labels['sha1sum'] == hash]
        label = label[self.configs].to_numpy()

        if self.transform:
            image = self.transform(image)

        return {'hash': hash, 'example': image, 'label': label}
