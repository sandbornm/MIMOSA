"""
Utilities for data wrangling the malware samples and other Mimosa data
into usable formats for ML
"""

import glob
import os
from os.path import join
import pandas as pd
import numpy as np
import cv2
from math import sqrt, ceil


def extract_binaries(pbitmap, psamples):
    """
    Extract sample binaries from subdirectories according to dataset defined in bitmap.
    """
    bins = glob.glob(psamples+'/**/*.bin', recursive=True)
    bitmap = pd.read_csv(pbitmap) if '.tsv' not in pbitmap else pd.read_csv(pbitmap, sep='\t')

    hashes = bitmap['sha1sum'].tolist()
    if not os.path.exists('bins'):
        os.makedirs('bins')

    missed = []
    for hash in hashes:
        found = False
        for bin in bins:
            if hash in bin:
                cmd = 'cp %s bins/%s.bin' % (bin, hash)
                os.system(cmd)
                found = True
                break

        if not found:
            missed += [hash]
            print('Sample not found: %s' % hash)

    res = os.listdir('bins')
    print('Total found =', len(res))
    return res


def samples2images(pbins):
    """
    Convert binary samples to byte arrays and save as square images.
    """
    bins = os.listdir(pbins)
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    dtype = np.dtype('B')
    min_width = None
    for bin in bins:
        hash, _ = os.path.splitext(os.path.basename(bin))
        try:
            with open(join(pbins, bin), "rb") as f:
                bytes = np.fromfile(f, dtype)
                width = int(ceil(sqrt(len(bytes))))  # get nearest greater square root
                bytes = np.hstack([bytes, np.zeros(width**2 - len(bytes), dtype)])
                if min_width is None or width < min_width:
                    min_width = width

                img = np.reshape(bytes, (width, width))  # create square image
                cv2.imwrite('imgs/%s.png' % hash, img)

        except IOError:
            print('Error opening file:', bin)

    print('Minimum image size =', min_width)


def rectify_df(df1,df2):
    hashes1 = df1['sha1sum'].tolist()
    hashes2 = df2['sha1sum'].tolist()

    minset = hashes1 if len(hashes1) < len(hashes2) else hashes2

    inds1 = []
    for i, hash in enumerate(hashes1):
        inds1 += [i] if hash not in minset else []

    inds2 = []
    for i, hash in enumerate(hashes2):
        inds2 += [i] if hash not in minset else []

    return df1.drop(inds1), df2.drop(inds2)


def create_bitmap_dataset(pbitmap, pevasion, bitmap_sz=15):
    bitmap = pd.read_csv(pbitmap) if '.tsv' not in pbitmap else pd.read_csv(pbitmap, sep='\t')
    evasion = pd.read_csv(pevasion) if '.tsv' not in pevasion else pd.read_csv(pevasion, sep='\t')

    # rectify hash mismatches
    bitmap, evasion = rectify_df(bitmap, evasion)
    bitmap, evasion = rectify_df(bitmap, evasion)

    ### X
    # create design matrix with entry for each bit
    bhashes = bitmap['sha1sum'].tolist()
    bits = bitmap[' bitmap'].astype(str).tolist()

    X = {i: [] for i in range(bitmap_sz)}
    X['hashes'] = bhashes
    for s in bits:
        leads = bitmap_sz - len(s)  # prepend leading zeros that CSV autotruncated
        map = '0'*leads + s
        for i, bit in enumerate(map):
            X[i] += [int(bit)]

    # save
    X_pd = pd.DataFrame.from_dict(X)
    X_pd.to_csv('bitmap_X.csv')

    X_np = X_pd.drop(columns=['hashes']).to_numpy()
    np.save('bitmap_X.npy', X_np)

    ### y
    # create label matrix with each config evasion vector
    configs = evasion.columns[1:]
    for config in configs:
        evasion[config] *= 1

    # save
    evasion.to_csv('bitmap_y.csv')

    y_np = evasion[configs].to_numpy()
    np.save('bitmap_y.npy', y_np)

    return X_np, y_np


if __name__ == '__main__':

    # # find binaries in sample dataset
    # fbitmap = 'hash_bitmap.tsv'
    # psamples = '../../../data/samples'
    # extract_binaries(fbitmap, psamples)

    # convert binary samples to images via numpy and cv2
    samples2images('bins')

    # # create bitmap dataset
    # fbitmap = 'hash_bitmap.tsv'
    # fevasion = 'fixed_evasion.csv'
    # X, y = create_bitmap_dataset(fbitmap, fevasion)
