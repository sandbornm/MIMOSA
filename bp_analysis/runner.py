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
import sys
import json


""" Class for running BluePill samples on specified backends using original MIMOSA configurations"""
class Runner:
    def __init__(self, sample_dir, config_dir):

        self.backends = ["vmware", "qemu", "kvm", "virtualbox"]
        assert isinstance(sample_dir, str) and os.path.isdir(sample_dir), "invalid sample directory"
        assert isinstance(config_file, str) and os.path.isfile(config_file), "invalid config file"

        self.samples = os.listdir(sample_dir)

    def load_configs(self):
        pass

    def attach_perf(self):
        pass

    def run(self):
        pass




if __name__ == "__main__":
    Runner().run()