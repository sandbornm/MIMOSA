import requests
import os
import sys
import json
from tqdm import tqdm

def run():
	# export VT_API_KEY as env variable
	API_KEY = os.environ['VT_API_KEY']
	SAMPLE_DIR = os.environ['SAMPLE_DIR']

	#query = sys.argv[1]
	#assert isinstance(query, str) and len(query) == 64 

	headers = {"Accept": "application/json",
			"X-Apikey": API_KEY}

	sample_list = os.listdir(SAMPLE_DIR)
	for sample in tqdm(sample_list):
		query = sample[:-4]
		# print(query)
		assert isinstance(sample, str) and len(sample) == 64
		url = f"https://www.virustotal.com/api/v3/files/{query}/behaviour_summary"
		response = requests.request("GET", url, headers=headers)
		report = response.text
		print(report)
		if "Invalid file hash" in report:
			print(f"{query} not valid")
		# 	print(report["error"])
		# fname = os.path.join(os.path.abspath(os.getcwd()), "fetched_reports", f"{query[:40]}-report.json")
		# with open(fname, "w") as f:
		# 	f.write(report)
		#print(f"wrote {fname}")

def read_reports():
	# with open(fname, "r") as f:
	# 	data = json.loads(f.read())
	# print(type(data))
	pass

if __name__ == "__main__":
	# todo take a file as input
	run()