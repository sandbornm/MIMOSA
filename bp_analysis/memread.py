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

import os
import logging
import subprocess
from parser import BehaviorReportParser


logging.basicConfig(filename='memread.log', level=logging.DEBUG)

""" view memory dumps from the vbox vm """
class MemReader:
    def __init__(self, artifacts_dir, sample_id, elf_file):
        assert os.path.exists(os.path.join(artifacts_dir, sample_id, elf_file))
        assert elf_file[-3:] == "elf"

        self.logger = logging.getLogger(__name__)

        self.artifacts_dir = artifacts_dir
        self.sample_id = sample_id
        self.elf_path = os.path.join(artifacts_dir, sample_id, elf_file)

        self.profile = "Win7SP1x86_23418"  # Win7SP0x86 was not working

        print("done init")

    # util to write bytes returned from subprocess calls to volatility to files
    def _write(self, fname, content):
        with open(os.path.join(self.artifacts_dir, self.sample_id, fname), 'wb') as f:
            f.write(content)


    def imagecopy(self):
        self.imcopy_raw = self.elf_path[:-4]+"_imcopy.raw"
        self.logger.info(f"running imagecopy with volatility to convert vbox mem dump {self.elf_path} to raw image {self.imcopy_raw}")
        subprocess.run(["python2", os.path.join(os.path.expanduser('~'), "volatility", "vol.py"), "-f", self.elf_path,
                          f"--profile={self.profile}", "imagecopy", "-O", self.imcopy_raw], stdout=subprocess.PIPE)
        print(f"ran imagecopy on {self.elf_path} and wrote {self.imcopy_raw}")

    def vboxinfo(self):
        self.logger.info(f"running vboxinfo with volatility to get details of [{self.elf_path}] vbox core dump")
        res = subprocess.run(["python2", os.path.join(os.path.expanduser('~'), "volatility", "vol.py"), "-f", self.elf_path,
                          f"--profile={self.profile}", "vboxinfo"], stdout=subprocess.PIPE)

        self._write(self.sample_id + "_vboxinfo", res.stdout)
        print(f"ran vboxinfo on {self.elf_path} and wrote result to {self.sample_id}_vboxinfo")

    # get list of running processes from the elf_path
    def pslist(self):
        self.logger.info(f"running pslist on {self.elf_path} to get running processes post-exec")
        res = subprocess.run(["python2", os.path.join(os.path.expanduser('~'), "volatility", "vol.py"),
                              "-f", self.elf_path, f"--profile={self.profile}", "pslist"], stdout=subprocess.PIPE)
        # vola -f <fname> --profile=self.profile pslist
        self._write(self.sample_id + "_pslist", res.stdout)
        print(f"ran pslist on {self.elf_path} and wrote {self.sample_id}_pslist")


if __name__ == "__main__":
    # memory analysis after trying to run samples
    brp = BehaviorReportParser()
    sample_id = "6d6d8bf6" # 6d6d8bf6, acdb3a93
    acdb_report = brp.get_report(sample_id)
    print(type(acdb_report))
    print(acdb_report['processes_created'])

    artifacts_dir = os.environ['ART_DIR']
    mr = MemReader(artifacts_dir, sample_id, f"{sample_id}_dump.elf")
    #mr.imagecopy()
    #mr.vboxinfo()
    mr.pslist()