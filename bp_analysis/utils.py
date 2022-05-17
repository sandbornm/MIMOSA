#!/usr/bin/env python3
# Copyright (C) 2022, Michael Sandborn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import glob
import os
import numpy as np
import cv2
from math import sqrt, ceil
from random import sample
import shutil

SAMPLE_DIR = os.environ['SAMPLE_DIR']

def samples2images(pbins):
    """
    Convert binary samples to byte arrays and save as square images -- hacked from Zach's work
    """
    bins = os.listdir(pbins)
    print(bins)
    # if not os.path.exists('imgs'):
    #     os.makedirs('imgs')

    dtype = np.dtype('B')
    min_width = None
    for bin in bins:
        hash, _ = os.path.splitext(os.path.basename(bin))
        try:
            with open(pbins+"/"+bin, "rb") as f:
                bytes = np.fromfile(f, dtype)
                width = int(ceil(sqrt(len(bytes))))  # get nearest greater square root
                bytes = np.hstack([bytes, np.zeros(width**2 - len(bytes), dtype)])
                if min_width is None or width < min_width:
                    min_width = width

                img = np.reshape(bytes, (width, width))  # create square image
                cv2.imwrite('bp_sample_imgs/%s.png' % hash, img)

        except IOError:
            print('Error opening file:', bin)

    print('Minimum image size =', min_width)


def write_txt(data, fname):
    fname = fname + ".txt"
    assert isinstance(data, list)
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cur_dir, fname), 'w') as f:
        f.write("\n".join(data))
    print(f"wrote {fname}")


def bp_samples_to_txt(sample_dir):
    bp_samples = os.listdir(sample_dir)
    write_txt(bp_samples, "bp_samples")


def bp_reports_to_txt(report_dir):
    bp_reports = os.listdir(report_dir)
    write_txt(bp_reports, "bp_reports")

# return a list of reports for samples given an abs path to text file
def bp_reports_from_text(fname):
    with open(fname, 'r') as reports_txt:
        lines = reports_txt.readlines()
        report_names = [line.rstrip() for line in lines]
    return report_names

# samples from the samples whose reports we have and move the binaries to dispatcher/samples/100_bluepill
def select_and_move_samples(fname, n=100, drc="100_bluepill"):
    chosen = [x.split('-')[0] for x in sample(bp_reports_from_text(fname), n)]

    bin_list = os.listdir(SAMPLE_DIR)
    report_dir = os.path.join(os.getcwd(), "bp_reports")
    report_list = os.listdir(report_dir)

    dst_root = os.path.join(os.path.dirname(os.getcwd()), "dispatcher", "samples", drc)
    if not os.path.exists(dst_root):
        print(f"creating directory {dst_root}")
        os.mkdir(dst_root)

    for c in chosen:
        samp_dir = os.path.join(dst_root, c)
        if not os.path.exists(samp_dir):
            print(f"creating directory {samp_dir}")
            os.mkdir(samp_dir)

        matched_bin = list(filter(lambda x: c in x, bin_list))[-1]
        matched_report = list(filter(lambda x: c in x, report_list))[-1]
        print(f"chosen is: {c} matched to {matched_bin} and {matched_report}")
        #assert len(src_bin) == 1 and len(src_report) == 1, "collision found"

        bin_src = os.path.join(SAMPLE_DIR, matched_bin)
        bin_dst = os.path.join(samp_dir, matched_bin[:40]+".bin")
        #print(f"moving sample {matched_bin[:40]} to {bin_dst}")
        shutil.copy(bin_src, bin_dst)  # copy binary

        report_src = os.path.join(report_dir, matched_report)
        report_dst = os.path.join(samp_dir, matched_report)
        #print(f"moving report {report_src} to {report_dst}")
        shutil.copy(report_src, report_dst)  # copy report


# pbins = os.environ['SAMPLE_DIR']
# bp_samples_to_txt(pbins)
# bp_reports_to_txt("/home/michael/MIMOSA/bp_analysis/bp_reports")
# samples2images(pbins)

select_and_move_samples(os.path.join(os.getcwd(), 'bp_reports.txt'))