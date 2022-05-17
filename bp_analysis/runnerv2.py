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
import subprocess
import logging
import time
import json

logging.basicConfig(filename='runner2.log', level=logging.DEBUG)


class Runnerv2:
    def __init__(self, conf):
        # todo update these with sample locations for each config
        self.sample_dir = os.environ['SAMPLE_DIR']
        self.report_dir = os.environ['REPORT_DIR']
        self.artifacts_dir = os.environ['ART_DIR']
        self.config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "config")
        self.exec_timeout = 30  # seconds to wait after executing sample
        self.conf = conf
        self.load_conf()

    def load_conf(self):
        for cf in os.listdir(self.config_dir):
            if cf == "win7x86.json":  # todo other windows versions
                with open(os.path.join(self.config_dir, cf)) as cfd:
                    config_data = json.load(cfd)
                    for k in config_data.keys():
                        if str(self.conf) in k:
                            conf_data = config_data[k]
        self.conf_data = conf_data if bool(conf_data) else None

    def make_results_dir(self):
        pass

    ################  Manage VM  ######################

    def set_vm_param(self, name, value):
        pass

    def set_vm_params(self):
        pass

    def clear_vm_params(self):
        pass

    def get_vm_param(self, name):
        pass

    def is_running(self):
        pass

    def start_image(self):
        pass

    def stop_image(self):
        pass

    def restore_orig_snapshot(self):
        pass

    def snapshot_vm(self):
        pass

    def resize_hd(self, new_size):
        pass

    ######################  Sample Execution  ######################

    def load_one_sample(self, sample):
        pass

    def run_sample_and_wait(self):
        pass

    def dump_core(self):
        pass

    def run_loop(self):
        pass


class VboxRunner(Runnerv2):
    def __init__(self, conf):
        self.name = "virtualbox"
        self.vm_name = "win7"
        self.vm_dir = os.environ["VBOX_DIR"]
        self.conf = conf
        super().__init__(conf)
        self.logger = logging.getLogger(f"{self.name}_runner")
        # todo add vbox sample locations
        self.samples = os.listdir(self.sample_dir)
        self.make_results_dir()
        self.conf_data = self.conf_data["extraConfig"][self.name]
        self.cur_sample_id = None  # the current sample executing

    def make_results_dir(self):
        self.res_path = os.path.join(self.artifacts_dir, f"{self.name}_runs")
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        print(f"created results dir at {self.res_path}")

    ################  Manage VM  ######################

    def set_vm_param(self, name, value):
        self.logger.info("[set_vm_param] setting {} to {}".format(name, value))
        subprocess.run(["vboxmanage", "setextradata", self.vm_name, name, value])

    def set_vm_params(self):
        print(f"{self.name} set vm params")
        self.logger.info("[set_vm_params] for {} configuration: {}".format(self.name, self.conf))
        for k, v in self.conf_data.items():
            self.set_vm_param(k, str(v))

    def get_vm_param(self, name):
        print(f"{self.name} get vm param")
        self.logger.info("[get_vm_param] getting {}".format(name))
        res = subprocess.run(["vboxmanage", "getextradata", self.vm_name, name], stdout=subprocess.PIPE)
        return str(res.stdout).split(":")[-1].replace('\n', '')

    def clear_vm_params(self):
        print(f"{self.name} clear vm params")
        self.logger.info("[clear_vm_params] for {} configuration: {}".format(self.name, self.conf))
        for k in self.conf_data:
            self.set_vm_param(k, "")

    def is_running(self):
        return self.vm_name in str(subprocess.check_output(["vboxmanage", "list", "runningvms"]))

    def start_image(self):
        print(f"start image {self.vm_name}")
        self.logger.info("[start_image] for {} vm {} config: {}".format(self.name, self.vm_name, self.conf))
        subprocess.run(["vboxmanage", "startvm", self.vm_name, "--type", "headless"])
        print("waiting for image start")
        time.sleep(20)

    def stop_image(self):
        print(f"stop image {self.vm_name}")
        self.logger.info("[stop_image] for vm {} config: {}".format(self.name, self.conf))
        subprocess.run(["vboxmanage", "controlvm", self.vm_name, "savestate"])  # try acpipowerbutton

    def restore_orig_snapshot(self):
        snapshot = "win7_bsnap_uac_off"  # base_no_uac_noGA"  # snapshot to reset to between executions
        print(f"restore orig snapshot from {snapshot} for {self.vm_name}")
        self.logger.info("[restore_snapshot] orig snapshot from {}".format(snapshot))
        print(f"restoring base snapshot for {self.vm_name}")
        subprocess.run(["vboxmanage", "snapshot", self.vm_name, "restore", snapshot])

    def snapshot_vm(self):
        snapshot = f"snap_post_{self.cur_sample_id}"
        print(f"taking snapshot {snapshot} of vm {self.vm_name}")
        self.logger.info("[snapshot_vm] taking snapshot {} of vm {}".format(snapshot, self.vm_name))
        subprocess.run(["vboxmanage", "snapshot", self.vm_name, "take", snapshot])

    def resize_hd(self, new_size):
        # todo check if size is valid
        print(f"resizing hd of {self.vm_name} to {new_size} (MB)")
        self.logger.info("[resize_hd] for {} to {} MB".format(self.vm_name, new_size))
        assert new_size % 10 == 0  # round MB values to resize
        subprocess.run(["vboxmanage", "modifymedium", self.vm_name, "--resize", int(new_size.strip())])

    ######################  Sample Execution  ######################

    def load_one_sample(self, sample):
        print(f"loading one sample {sample[:8]}")
        self.logger.info("[load_sample] for configuration: {}".format(self.conf_data))
        subprocess.run(["vboxmanage", "controlvm", self.vm_name, "setcredentials", "hive", "hive", "TEST"])
        self.cur_sample_id = sample[:8]
        assert self.cur_sample_id is not None
        subprocess.run(
            ["vboxmanage", "guestcontrol", self.vm_name, "--username", "hive", "--password", "hive", "copyto",
             os.path.join(self.sample_dir, sample), f"C:\\Users\\hive\\{self.cur_sample_id}.exe"])
        print(f"waiting on sample {sample[:8]} load...")
        time.sleep(20)

    def run_sample_and_wait(self):
        assert self.cur_sample_id is not None
        print(f"run and wait for sample {self.cur_sample_id}")
        self.logger.info("attempting execution of sample {}".format(self.cur_sample_id))
        subprocess.Popen(["vboxmanage", "guestcontrol", self.vm_name, "--username", "hive",
                          "--password", "hive", "run", "--", f"C:\\Users\\hive\\{self.cur_sample_id}.exe"])
        self.logger.info("starting wait on sample {}".format(self.cur_sample_id))
        time.sleep(self.exec_timeout)
        print(f"timeout reached for sample {self.cur_sample_id}")

    def dump_core(self):
        assert self.cur_sample_id is not None
        print(f"dump core of vm {self.vm_name} after executing {self.cur_sample_id}")
        self.logger.info("dumping RAM of {} after running sample {}".format(self.vm_name, self.cur_sample_id))
        subprocess.Popen(["vboxmanage", "debugvm", self.vm_name, "dumpvmcore", "--filename",
                          os.path.join(self.res_path, self.cur_sample_id, self.cur_sample_id + "_dump.elf")])
        print(f"RAM dumped to {self.cur_sample_id + '_dump.elf'}")

    def run_loop(self):
        samples_run = []
        for sample in self.samples[:1]:  # just two for now

            cur_sample = sample[:10]
            print(f"running sample: {cur_sample}")

            if not self.is_running():
                self.restore_orig_snapshot()  # restore from clean install
                self.clear_vm_params()  # clear vm params to avoid uuid not found err
                self.start_image()  # start vm at clean install

            self.set_vm_params()  # set vm-specific params, reset after execution
            self.load_one_sample(sample)  # set the current sample to be executed
            self.run_sample_and_wait()  # call exec on the copied sample file
            self.dump_core()  # store a dump of memory for later analysis with volatility
            self.clear_vm_params()  # clear the vm_params from the configuration (VM UUID error if dont do this)
            self.snapshot_vm()  # snapshot the vm after the sample executes to save for later
            self.stop_image()  # save the state of the image / also consider poweroff over savestate?
            print(f"waiting for shutdown after running sample: {cur_sample}")
            time.sleep(10)  # hack to wait for shutdown
            samples_run.append(self.cur_sample_id)
        print("analysis done")
        print(f"samples run: {samples_run}")
        return samples_run


class VmwareRunner(Runnerv2):
    name = "vmware"

    def __init__(self, conf):
        self.name = "vmware"
        self.conf = conf
        super().__init__(conf)
        self.logger = logging.getLogger(f"{self.name}_runner")
        self.make_results_dir()
        self.conf_data = self.conf_data["extraConfig"][self.name]

    def make_results_dir(self):
        res_path = os.path.join(self.artifacts_dir, f"{self.name}_runs")
        if not os.path.exists(res_path):
            os.mkdir(res_path)



class QemuRunner(Runnerv2):
    def __init__(self, conf):
        self.name = "kvm"  # kvm or qemu??
        self.conf = conf
        super().__init__(conf)
        self.logger = logging.getLogger(f"{self.name}_runner")
        self.make_results_dir()
        self.conf_data = self.conf_data["extraConfig"][self.name]

    def make_results_dir(self):
        res_path = os.path.join(self.artifacts_dir, f"{self.name}_runs")
        if not os.path.exists(res_path):
            os.mkdir(res_path)

