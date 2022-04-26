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

import os
import matplotlib.pyplot as plt
from parser import BehaviorReportParser
import time

report_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bp_reports")
bpr = BehaviorReportParser(report_dir)

# set of all keys used across all reports
keys = bpr.get_all_keys()

# list of tuples of (sample_id, num keys in report dict, total num vals in report dict)
sample_desc = bpr.report_summary()

# list the keys for a specific sample by passed ID (first 40 chars of sample sum)
sample_keys = bpr.get_report_keys("0a73238e2b62082efe1bd20f4e7904da1ea15678")
print(sample_keys)

# visualize the number of keys and values included in the report of each sample
# as a rough indicator of how much information is present across all reports
# and to help decide which samples are worth studying further
def key_count_histogram(key_counts, val_counts):
    assert isinstance(key_counts, list) and isinstance(val_counts, list)
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(key_counts, label="key counts", bins=8)
    axs[0].set_xlabel("# keys in report")
    axs[1].hist(val_counts, color='r', label="value counts")
    axs[1].set_xlabel("# values in report")
    plt.ylabel("count")
    fig.suptitle(f"key, value counts for BluePill behavior reports (n={len(key_counts)})")

    fname = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "fig", f"bp_counts_{time.strftime('%Y%m%d-%H%M%S')}")
    plt.savefig(fname)
    #plt.show()


if __name__ == "__main__":
    key_counts = [data[1] for data in sample_desc]
    value_counts = [data[2] for data in sample_desc]
    key_count_histogram(key_counts, value_counts)

