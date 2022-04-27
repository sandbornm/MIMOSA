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
import subprocess
import logging


logging.basicConfig(filename='runner.log', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.debug("hello from debug")
log.info("hello from info")
log.warning("hello from warning")
log.error("hello from error")


""" Class for running BluePill samples on specified backends using original MIMOSA configurations"""
class Runner:
    def __init__(self, sample_dir, config_dir, config="win7x86_conf1", host_ip="8.8.8.8"):

        self.backends = ["vmware", "qemu", "kvm", "virtualbox"]
        self.default_backend = "virtualbox"
        
        assert isinstance(sample_dir, str) and os.path.isdir(sample_dir), "invalid sample directory"
        assert isinstance(config_dir, str) and os.path.isdir(config_dir), "invalid config file"
        self.sample_dir = sample_dir
        self.config_dir = config_dir
        
        self.samples = os.listdir(self.sample_dir)
        self.config_files = os.listdir(self.config_dir)

        self.all_configs = ["win7x86_conf1", "win7x86_conf2", "win7x86_conf3", "win7x86_conf4"]
        
        assert config in self.all_configs, "invalid config specified"
        self.config = config # focus on one for now
        
        # todo costly in time to create VMs from runner
        # self.config_dict = self.load_configs()
    
        # check if there is already a vdi image present
        self.vdi_present = self.vdi_exists()


    def vdi_exists(self):
        return os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "img", f"{self.config}.vdi"))

    def load_configs(self):
        print("load_configs()")
        config_data = {}
        for cf in self.config_files:
            print(cf)
            if cf == "win7x86.json":
                with open(os.path.join(self.config_dir, cf)) as cfd:
                    config_data = json.load(cfd)
                    print(type(config_data), config_data)
        return config_data if bool(config_data) else None

    def attach_monitors(self):
        # add volatility or other monitor... also using subprocess?
        from volatility3.framework import interfaces
        
        subprocess.run(["./scripts/attach_monitors.sh"])

    def start_image(self):
        print("start_image()")
        # subprocess call to vmcloak to python2? is possible?
        subprocess.run(["echo", "start image!"])

    def stop_image(self):
        print("stop_image()")
        subprocess.run(["echo", "stop image!"])
        # subrocess call to vmcloak
        
    def vmcloak_ops(self):
        print("vmcloak_ops()")
        subprocess.run(["./scripts/vmcloak_ops.sh"])

    def mw_exec(self, sample):
        print("mw_exec()")
        assert isinstance(sample, str) and len(sample) == 64
        # calls  mw_exec.sh script which places the current
        # sample into the vm and runs it
        subprocess.run(["./scripts/mw_exec.sh"])


    def detox(self):
        print("detox()")


    def run(self):
        for sample in self.samples[:1]: # just one for now
            print(f"running sample: {sample[:40]}")
            self.attach_monitors()
            self.vmcloak_ops()
            self.start_image()
            self.mw_exec(sample)
            self.stop_image()





if __name__ == "__main__":
    sample_dir = os.environ["SAMPLE_DIR"]
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "config")
    r = Runner(sample_dir, config_dir)
    r.run()