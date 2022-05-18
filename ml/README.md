# Mimosa-ML
Zach Stoebner

Leveraging ML to effectively scale dynamic malware analysis. Specifically, a deep learning multilabel classifier maps a stealthy malware sample to a set of configurations under which it is most likely to run. 

[comet](https://www.comet.ml/zstoebs/mimosa/view/new/panels)


## Main Scripts
- [Bitmap2Config](/Bitmap2Config.py) implements multilabel classification of a pre-labeled sample bitmap of which artifacts that it detects to a config bitmap. This technique utilizes a perceptron with 20 neurons with near perfect exact match ratio and recall. However, the pre-labeling prerequisite on the domain imposes non-generalizable, and likely unrealistic, applicability. *This scenario primarily serves as a proof-of-concept.*

- [run](/run.py) implements multilabel classification of a sample binary to a config bitmap. The binary can be represented by two modalities: 1. byte sequence and 2. byte image. The former leverages sequence models while the latter leverages vision models. Intuitively, sequence models recognize patters in subsequent instructions and the order in which they occur, which necessarily depends on their location in the binary. On the other hand, vision models recognize patterns of code blocks, regardless of neighborhood in the raw binary. *This scenario is more generalizable since the domain is less constrained, solely requiring the malware binary.*

- [Search](/raytune.py) leverages RayTune to implement a continuous random hyperparameter search. *The models, especially for vision, can be upwards of 20 GiB which can cause CUDA OOM errors, particularly if running on a single typical GPU with 12 GiB vRAM.*


## Setup
`conda env create -f no-builds.yml`	

## Usage
[run](/run.py) is the main POI for most users, simplifying the ML workflow to arguments passed to a single command. 

Understanding the ML workflow as 4 modes: 
1. Training = learn weights and biases for a model with paired data; *must specify paths to examples and labels*
2. Cross-validation = train multiple models on the dataset with a different validation set each time to quantify performance; *must specify paths to examples and labels*
3. Testing = test a trained model on a test set and compute scores; *must specify paths to examples, labels, and a saved model*
4. Prediction = predict results for examples without labels; *must specify paths to examples and a saved model*

- Help: `python run.py -h`
- Refer to [run.sh](/run.sh) for a shell script that outlines the command and all useful arguments. Refer to [util](/util/__init__.py) for the detailed implementation of the argument parser. 
