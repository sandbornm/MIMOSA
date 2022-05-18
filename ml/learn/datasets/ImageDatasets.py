from os.path import join, basename, splitext
import torch
import cv2
import glob
import pandas as pd
from torch.utils.data import Dataset


class MalwareImageDataset(Dataset):
    def __init__(self, args):
        """
        Dataset class for images of malware binaries for deep learning multilabel classification
        """
        self.mode = args['mode']
        self.examples_dir = args['examples_dir']
        self.sz = args['size']
        self.transform = args['transform']

        self.examples = glob.glob(join(self.examples_dir, '*.png'))
        self.hashes = [splitext(basename(ex))[0] for ex in self.examples]

        self.n_examples = len(self.examples)
        self.n_classes = args['n_classes']
        self.classes = range(self.n_classes)

        if args['labels_csv'] and not self.mode.lower() == 'predict':
            self.labels_csv = args['labels_csv']
            self.labels = pd.read_csv(self.labels_csv) if '.tsv' not in self.labels_csv else pd.read_csv(
                self.labels_csv, sep='\t')
            self.classes = self.labels.columns[1:]

            n_labels = len(self.labels)
            n_label_cols = len(self.classes)
            assert n_labels == self.n_examples, 'Number of labels (%d) does not equal number of examples (%d)' % (n_labels, self.n_examples)
            assert n_label_cols == self.n_classes, 'Label length (%d) does not equal number of classes (%d)' % (n_label_cols, self.n_classes)

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
        label = label[self.classes].to_numpy()

        if self.transform:
            image = self.transform(image)

        return {'hash': hash, 'example': image, 'label': label}
