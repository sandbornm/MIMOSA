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
import subprocess
import logging
import time
import json

logging.basicConfig(filename='runner.log', level=logging.DEBUG)


""" run BluePill samples on specified backends (vbox to start) using (1 for now) original MIMOSA configurations

    sample_dir: folder containing the samples to study (dir)
    config_dir: folder specifying the VM configurations as json (dir)
    artifacts_dir: folder to place results from analysis (dir)
     
    config: the name of the configuration to study (str)
    
    exec_timeout: # seconds to wait after running exec until postprocess

"""
class Runner:
    def __init__(self, sample_dir, config_dir, artifacts_dir, config="win7x86_conf1", exec_timeout=30):

        self.logger = logging.getLogger(__name__)
        self.backends = ["vmware", "qemu", "kvm", "virtualbox"]
        self.default_backend = "virtualbox"
        self.backend = self.default_backend
        
        assert isinstance(sample_dir, str) and os.path.isdir(sample_dir), "invalid sample directory"
        assert isinstance(config_dir, str) and os.path.isdir(config_dir), "invalid config file"

        self.sample_dir = sample_dir
        self.config_dir = config_dir
        self.artifacts_dir = artifacts_dir

        self.config_files = os.listdir(self.config_dir)
        self.all_configs = ["win7x86_conf1", "win7x86_conf2", "win7x86_conf3", "win7x86_conf4"]
        assert config in self.all_configs, "invalid config specified"
        self.config = config
        self.config_dict = self.load_configs()

        self.vm_dir = os.environ['VM_DIR']
        self.vm_name = "win7"
        self.exec_timeout = exec_timeout # seconds until abort sample
        self.samples = os.listdir(self.sample_dir)
        self.cur_sample = None

    """ convert the json config files into a dict """
    def load_configs(self):
        self.logger.info("loading configs")
        config_data = {}
        for cf in self.config_files:
            if cf == "win7x86.json":
                with open(os.path.join(self.config_dir, cf)) as cfd:
                    config_data = json.load(cfd)
        return config_data if bool(config_data) else None


    """ create a folder for each sample to track the artifacts from running the sample """
    def make_results_dir(self):
        if not os.path.isdir(os.path.join(self.artifacts_dir, self.cur_sample_id)):
            os.mkdir(os.path.join(self.artifacts_dir, self.cur_sample_id))

    """ Set VM parameters as specified in the extraConfig section of the config dict """
    def set_vm_param(self, key, val):
        self.logger.info("[set_vm_param] setting {} to {}".format(key, val))
        subprocess.run(["vboxmanage", "setextradata", self.vm_name, key, val])

    def get_vm_param(self, key):
        self.logger.info("[get_vm_param] getting {}".format(key))
        res = subprocess.run(["vboxmanage", "getextradata", self.vm_name, key], stdout=subprocess.PIPE)
        ret = str(res.stdout).split(":")[-1].replace('\n', '')
        return ret

    def configure_vm_params(self):
        self.logger.info("[configure_vm_params] for configuration: {} and backend: {}".format(self.config, self.backend))
        vars_to_set = self.config_dict[self.config]["extraConfig"][self.backend]
        for k, v in vars_to_set.items():
            self.set_vm_param(k, str(v))

    def clear_vm_params(self):
        self.logger.info("[clear_vm_params] for configuration: {} and backend: {}".format(self.config, self.backend))
        vars_to_clear = self.config_dict[self.config]["extraConfig"][self.backend]
        for k in vars_to_clear:
            self.set_vm_param(k, "")
        print("VM params cleared")



    """ VM ops """
    def get_running_vms(self):
        return subprocess.check_output(["vboxmanage", "list", "runningvms"])

    def is_running(self):
        return self.vm_name in str(self.get_running_vms())

    def start_image(self):
        self.logger.info("[start_image] for configuration: {}".format(self.config))
        print("starting headless VM from base snapshot")
        subprocess.run(["vboxmanage", "startvm", self.vm_name, "--type", "headless"])
        print("waiting for image start")
        time.sleep(20) # hack for waiting

    def stop_image(self):
        self.logger.info("[stop_image] for configuration: {}".format(self.config))
        subprocess.run(["vboxmanage", "controlvm", self.vm_name, "savestate"])

    def snapshot_vm(self):
        self.logger.info("[snapshot_vm] for configuration: {}".format(self.config))
        subprocess.run(["vboxmanage", "snapshot", self.vm_name, "take", self.cur_sample_id+"_post"])

    def install_guest_additions(self):
        self.logger.info("[install_guest_additions] for configuration: {}".format(self.config))
        subprocess.run(["vboxmanage", "guestcontrol", "updateadditions",
                        self.vm_name, "--source", "/usr/share/virtualbox/VBoxGuestAdditions.iso"])

    def restore_orig_snapshot(self):
        snapshot = "win7_bsnap_uac_off"
        self.logger.info("[restore_snapshot] orig snapshot from {}".format(snapshot))
        print("restoring base snapshot")
        subprocess.run(["vboxmanage", "snapshot", self.vm_name, "restore", snapshot])


    """ handle samples """
    def load_samples_folder(self):
        self.logger.info("[load_samples_folder] for configuration: {}".format(self.config))
        subprocess.run(["vboxmanage", "sharedfolder", "add",
                        self.vm_name, "--name", "bp_samples", "--hostpath", self.sample_dir])

    def load_sample(self, sample):
        self.logger.info("[load_sample] for configuration: {}".format(self.config))
        subprocess.run(["vboxmanage", "controlvm", self.vm_name, "setcredentials", "hive", "hive", "TEST"])
        self.cur_sample = sample
        self.cur_sample_id = self.cur_sample[:8]
        subprocess.run(["vboxmanage", "guestcontrol", self.vm_name, "--username", "hive", "--password", "hive", "copyto",
                        os.path.join(self.sample_dir, self.cur_sample), f"C:\\Users\\hive\\{self.cur_sample_id}.exe"])

    def run_sample(self):
        self.logger.info("attempting execution of sample {}".format(self.cur_sample_id))
        print(f"start exec sample {self.cur_sample_id}")
        subprocess.Popen(["vboxmanage", "guestcontrol", self.vm_name, "--username", "hive",
                        "--password", "hive", "run", "--", f"C:\\Users\\hive\\{self.cur_sample_id}.exe"])

    def wait_sample(self):
        self.logger.info("starting wait on sample {}".format(self.cur_sample_id))
        time.sleep(self.exec_timeout)
        print("done waiting")

    """ get memory information from running vm """
    def dump_ram(self):
        self.logger.info("dumping RAM after running sample {}".format(self.cur_sample_id))
        subprocess.Popen(["vboxmanage", "debugvm", self.vm_name, "dumpvmcore", "--filename",
                        os.path.join(self.artifacts_dir, self.cur_sample_id, self.cur_sample_id+"_dump.elf")])
        print(f"RAM dumped to {self.cur_sample_id+'_dump.elf'}")

    """ obtain process information before and after execution with the tasklist.exe tool"""
    def get_tasklist(self, before=True):
        if before:
            self.logger.info("getting tasklist before running sample {}".format(self.cur_sample_id))
        else:
            self.logger.info("getting tasklist after running sample {}".format(self.cur_sample_id))
        res = subprocess.run(["vboxmanage", "guestcontrol", self.vm_name, "--username", "hive",
                        "--password", "hive", "run", "--", "tasklist.exe"], stdout=subprocess.PIPE)

        def write(fname):
            with open(os.path.join(self.artifacts_dir, self.cur_sample_id, fname), 'wb') as f:
                f.write(res.stdout)

        if before:
            write(self.cur_sample_id+"_ps_pre")
            print("wrote ps pre to file")
        else:
            write(self.cur_sample_id+"_ps_post")
            print("wrote ps post to file")


    def run(self):
        samples_run = []
        for sample in self.samples[:2]: # just two for now

            print(f"running sample: {sample[:10]}")

            if not self.is_running():
                self.restore_orig_snapshot()  # restore from clean install
                self.start_image()  # start vm at clean install

            self.load_sample(sample)  # load a binary sample
            self.make_results_dir()  # create a folder to store artifacts
            self.get_tasklist()  # list processes running pre-execution
            self.configure_vm_params()  # set vm params from extraConfig of json config
            self.run_sample()  # call exec on the copied sample file
            self.wait_sample()  # wait exec_timeout seconds for the sample to run
            self.get_tasklist(before=False)  # get the process list after sample has chance to be nasty
            self.dump_ram()  # store a dump of memory for later analysis with volatility
            self.clear_vm_params()  # clear the vm_params from the configuration (VM UUID error if dont do this)
            self.snapshot_vm()  # snapshot the vm after the sample executes to save for later
            self.stop_image()  # save the state of the image / also consider poweroff over savestate?
            time.sleep(10)  # hack to wait for shutdown
            samples_run.append(self.cur_sample_id)
        print("analysis done")
        print(f"samples run: {samples_run}")
        return samples_run



if __name__ == "__main__":
    sample_dir = os.environ["SAMPLE_DIR"]
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "config")
    artifacts_dir = os.environ["ART_DIR"]

    r = Runner(sample_dir, config_dir, artifacts_dir)
    r.clear_vm_params()  # precautionary; UUID Error thrown with extraConfig strings set
    run_list = r.run()  # try to run 2 samples

