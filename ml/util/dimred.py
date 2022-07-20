"""
Dimensionality reduction on sample binaries
"""

import os
import argparse
import numpy as np
from sklearn.manifold import TSNE


def get_args():
    parser = argparse.ArgumentParser(description='Dim reduction args')

    parser.add_argument('--p_bins', '-p', type=str, required=True, help='path to dir containing sample bins')

    return parser.parse_args()


def main(args):
    return


if __name__ == '__main__':
    args = get_args()

    main(args)
