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

import argparse
import os
import sys
import runnerv2
# todo import runners

def run(args):
    if args.runner_kind == "vbox":
        vbr = runnerv2.VboxRunner(args.conf)
        vbr.run_loop()
    elif args.runner_kind == "vmware":
        vmr = runnerv2.VmwareRunner(args.conf)
    elif args.runner_kind == "qemu":
        qmr = runnerv2.QemuRunner(args.conf)

def validate_args(args):
    assert args.runner_kind in ["vbox", "vmware", "qemu"], f"{args.runner_kind} not a valid runner"

    if args.runner_kind == "vbox" or args.runner_kind == "qemu":
        assert args.conf == 1, f"{args.conf} not a valid conf for this runner"
    elif args.runner_kind == "vmware":
        assert args.conf == 3, f"{args.conf} not a valid conf for this runner"
    # ["vmware_conf3", "qemu_legacy_conf1", "vbox_conf1"]
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("runner_kind", type=str)
    parser.add_argument("conf", type=int)

    args = parser.parse_args()

    if validate_args(args):
        run(args)
    else:
        print("invalid args, exiting")
        sys.exit(0)

    run(args)
