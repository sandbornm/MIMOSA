import requests
import os
import sys


def run():
	# export VT_API_KEY as env variable
	API_KEY=os.environ['VT_API_KEY']

	query = sys.argv[1]
	assert isinstance(query, str) and len(query) == 64 

	headers = {"Accept": "application/json",
			"X-Apikey": API_KEY}

	url = f"https://www.virustotal.com/api/v3/files/{query}/behaviour_summary"
	response = requests.request("GET", url, headers=headers)
	report = response.text.data

	print(response.text)

if __name__ == "__main__":
	# todo take a file as input
	run()