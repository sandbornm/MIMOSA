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


import requests
import os
import json
from tqdm import tqdm

""" Fetch json of malware behavior reports given a VirusTotal api key
    and a path to the samples of interest as environment variables
"""


class VTReportFetcher:
    def __init__(self, report_dir):
        assert 'VT_API_KEY' in os.environ, "missing VT_API_KEY!"
        assert 'SAMPLE_DIR' in os.environ, "missing SAMPLE_DIR!"
        assert os.path.isdir(report_dir), "report_dir is not a directory!"

        self.API_KEY = os.environ['VT_API_KEY']
        # path to dir of samples named by length 64 sums (bp_reports named by first 40 chars)
        self.SAMPLE_DIR = os.environ['SAMPLE_DIR']

        self.report_dir = report_dir

        self.headers = {"Accept": "application/json", "X-Apikey": self.API_KEY}

        self.missing_reports = []

        self.samples = os.listdir(self.SAMPLE_DIR)
        self.num_samples = len(self.samples)

    def fetch(self):

        missing_reports = []
        for sample in tqdm(self.samples[815:1000]):  # 500 : 600
            # if "6d6d8bf6" == sample[:8]:
            #     print("found 6d6d")
            assert len(sample) == 64, f"invalid file name {sample}"
            url = f"https://www.virustotal.com/api/v3/files/{sample}/behaviour_summary"
            response = requests.request("GET", url, headers=self.headers)

            if response.status_code != 200:
                self.missing_reports.append(sample)

            report = json.loads(response.text)
            assert isinstance(report, dict)
            if 'error' in report and report['error']['message'] == "Quota exceeded":
                print(f"quota exceeded {sample}")
                break  # probably an error such as not found or quota exceeded
            else:
                fname = os.path.join(self.report_dir, f"{sample[:40]}-report.json")
                if fname not in os.listdir(self.report_dir):
                    with open(fname, 'w') as report_file:
                        json.dump(response.json(), report_file)

        with open(os.path.join(os.getcwd(), "missing_reports.txt"), 'a') as missing:
            for mr in missing_reports:
                missing.write(f"{mr}\n")

        print(f"{len(missing_reports)} bp_reports missing")


# sanity check that the reports were pulled properly from VT
def find_and_rm_error_reports(report_dir):
    for report_file in os.listdir(report_dir):
        with open(os.path.join(report_dir, report_file)) as f:
            report = json.load(f)
            if "error" in report.keys():
                print(f"bad report in {report_file}! Error: {report['error']}")
                os.remove(os.path.join(report_dir, report_file))
                print(f"removed {report_file} due to error")
            elif not report["data"]:
                print(f"empty report in {report_file}: {report['data']}")
                os.remove(os.path.join(report_dir, report_file))
                print(f"removed {report_file} due to emptiness")

# def find_and_rm_quota_exceeded(report_dir):
#     for report_file in os.listdir(report_dir):
#         with open(os.path.join(report_dir, report_file)) as f:
#             report = json.load(f)
#             if "error" in report.keys() and report["error"]["message"] == "Quota exceeded":
#                 #  remove this report, it contains nothing useful


if __name__ == "__main__":
    report_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bp_reports")
    print(f"report file count before filtering: {len(os.listdir(report_dir))}")
    VTReportFetcher(report_dir).fetch()
    find_and_rm_error_reports(report_dir)
    print(f"report file count after filtering: {len(os.listdir(report_dir))}")
