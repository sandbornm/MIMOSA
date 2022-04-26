# Mimosa-ML
Zach Stoebner

Leveraging ML to effectively scale dynamic malware analysis. Specifically, a deep learning multilabel classifier maps a stealthy malware sample to a set of configurations under which it is most likely to run. 

[comet](https://www.comet.ml/zstoebs/mimosa/view/new/panels)

## Main Scripts
- [Bitmap2Config](/Bitmap2Config.py): multilabel classification of a pre-labeled sample bitmap of which artifacts that it detects to a config bitmap. 

- [Malware2Config](/Malware2Config.py): multilabel classification of a PNG image of the sample binary to a config bitmap. 

- [Search](/search.py): leverages RayTune to conduct an extensive hyperparameter search. 

## Usage
1. Set up conda environment: `conda env create -f environment.yml`
2. Run main script with appropriate args. 
	- *Ensure that experiment in learn/__init__.py points to correct comet project.*
