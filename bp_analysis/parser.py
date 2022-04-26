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
import json
from tqdm import tqdm

class BehaviorReportParser:

    # CPU, HCI, Process, Storage, Network
    def __init__(self, report_dir):
        assert os.path.isdir(report_dir), "report_dir is not a directory!"
        self.report_dir = report_dir
        self.reports = [os.path.join(report_dir, x) for x in os.listdir(report_dir)]
        self.num_reports = len(self.reports)

        """
        'memory_pattern_ips', 'processes_terminated', 'hosts_file', 'services_stopped',
        'dns_lookups', 'services_created', 'services_opened', 'processes_tree', 'files_deleted',
        'signals_hooked', 'processes_injected', 'memory_pattern_domains', 'verdict_labels',
        'files_copied', 'windows_searched', 'registry_keys_deleted', 'files_opened',
        'files_attribute_changed', 'modules_loaded', 'http_conversations', 'permissions_requested',
        'text_highlighted', 'tags', 'ip_traffic', 'mutexes_created', 'processes_created', 'mutexes_opened',
        'files_dropped', 'files_written', 'registry_keys_opened', 'verdicts', 'verdict_confidence',
        'services_started', 'command_executions', 'registry_keys_set', 'processes_killed'
        
        
        important: command_executions, files_opened, processes_created, ip_traffic, verdicts
        
        -- search for evasive?
        """
        self.all_keys = set()
        self.all_data = {}

        for report_file in tqdm(self.reports):
            with open(report_file) as f:
                report = json.load(f)
                for k in report["data"].keys():
                    self.all_keys.add(k)
                self.all_data[report_file.split("/")[-1].split("-")[0]] = report["data"]

    def get_all_keys(self):
        #print(self.report_keys)
        #print(len(self.report_keys))
        return self.all_keys

    def get_report_keys(self, report_id):
        assert report_id in self.all_data.keys(), "report not found!"
        return list(self.all_data[report_id].keys())

    def report_summary(self):
        sample_desc = []
        for sample, report in self.all_data.items():
            num_keys = len(list(report.keys()))
            num_values = sum([len(v) for v in list(report.values()) if isinstance(v, list)])
            print(f"{sample} : {num_keys} keys with {num_values} total values")
            sample_desc.append((sample, num_keys, num_values))
        return sample_desc


if __name__ == "__main__":
    report_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bp_reports")
    BehaviorReportParser(report_dir).report_summary()
