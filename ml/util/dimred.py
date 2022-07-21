"""
Dimensionality reduction on sample binaries
"""

import os
from os.path import join
import argparse
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Dim reduction args')

    parser.add_argument('--p_bins', '-p', type=str, nargs='+', required=True, help='path to dir(s) containing sample bins')
    parser.add_argument('--dim_embed', '-de', type=int, default=3, help='dimension of the embedding')
    parser.add_argument('--perplexity', '-pe', type=float, default=30.0, help='dimension of the embedding')

    return parser.parse_args()


def create_dataset(bin_dirs):
    """
    Creates a dataset given a list of dirs containing binary samples

    Returns:
        datas = stacked dataset
        classes = list of class names and their lengths
    """
    print('Creating dataset...')

    # create dataset
    classes = []
    datas = []
    mnmn = None  # use shortest byte vector length as dim of all examples in all classes
    for dir in bin_dirs:
        cls = os.path.basename(dir)
        bins = os.listdir(dir)
        n_bins = len(bins)
        classes += [(cls, n_bins)]

        print('Processing %d bins for class %s' % (n_bins, cls))

        dtype = np.dtype('B')
        data = None
        mn = None
        for bin in bins:
            hash, _ = os.path.splitext(os.path.basename(bin))
            try:
                with open(join(dir, bin), "rb") as f:
                    bytes = np.fromfile(f, dtype).reshape(1, -1)

                    l = bytes.shape[-1]
                    if not mn:
                        data = bytes
                        mn = l
                    elif l < mn:  # new one is shortest
                        mn = l
                        data = np.vstack([data[:, :mn], bytes])
                    else:  # bytes is not shortest so far
                        data = np.vstack([data, bytes[:, :mn]])

            except IOError:
                print('Error opening file:', bin)

        print('Min for class % s: ' % cls, mn)

        mnmn = mn if not mnmn or mn < mnmn else mnmn
        datas += [data]

    print('Overall min: ', mnmn)
    datas = np.vstack([data[:, :mnmn] for data in datas])
    print('Dataset shape: ', datas.shape)

    return datas, classes


def dimred(data, dim_embed, perp):
    """
    Dimensionality reduction on different classes of binary samples
    """
    print('Performing dim reduction...')

    tsne = TSNE(n_components=dim_embed, perplexity=perp)
    return tsne.fit_transform(data)


def visualize(data, names, dim_embed, **kwargs):
    """
    Visualize dim reduced data
    """
    print('Visualizing...')
    add = []
    for key, item in kwargs.items():
        add += [str(key) + '=' + str(item)]

    if not os.path.exists('results'):
        os.makedirs('results')
        print('Created results dir')

    if dim_embed == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for name, d in zip(names, data):
            ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=name)

        ax.legend()
        # plt.show()
        fname = join('results', '_'.join(names + add) + '_3d.png')
        fig.savefig(fname)
        print('Saved to ', fname)
    elif dim_embed == 2:
        return
    else:
        print('Unable to visualize dim %d. Only 2D or 3D available.' % dim_embed)


if __name__ == '__main__':
    args = get_args()

    bin_dirs = args.p_bins
    dim_embed = args.dim_embed
    perp = args.perplexity

    # compose each class into a single high-dim array and perform dim reduction
    data, classes = create_dataset(bin_dirs)
    dr_data = dimred(data, dim_embed, perp)

    # split dataset back into list of arrays indexed by class
    split_inds = []
    names = []
    for name, length in classes:
        ind = length if not split_inds else split_inds[-1] + length
        split_inds += [ind]
        names += [name]

    dr_data = np.split(dr_data, split_inds)

    # visualize the dim reduction
    visualize(dr_data, names, dim_embed, perp=perp)
