# Mimosa-ML
Zach Stoebner

## Main
- [Bitmap2Config](/Bitmap2Config.py): multilabel classification of a pre-labeled sample bitmap of which artifacts that it detects to a config bitmap. 

- [Malware2Config](/Malware2Config.py): multilabel classification of a PNG image of the sample binary to a config bitmap. 

## Usage
1. Set up conda environment: `conda env create -f environment.yml`
2. Run main script with appropriate args. 
	- *Ensure that experiment in Malware2Config points to correct comet project.*
