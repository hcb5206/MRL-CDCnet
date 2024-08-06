# Project Name
Difference Construct Connecting Bond of Consistency and Complementarity to Enhanced Multimodal Representation Learning

## Introduction
This document provides the PyTorch implementation of CDCnet. There are a total of six projects corresponding to experiments on six datasets, with each project named after its corresponding dataset.

## Data
The original data for the six datasets used in this study can be found at: [IEMOCAP](https://sail.usc.edu/iemocap/), 
[MELD](https://github.com/declare-lab/MELD), [CMU-MOSEI](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK), 
[Twitter2019](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection?tab=readme-ov-file), 
[CrisisMMD](https://crisisnlp.qcri.org/crisismmd), 
[DMD](https://archive.ics.uci.edu/dataset/456/multimodal+damage+identification+for+humanitarian+computing). 
However, we recommend using the pre-extracted feature data we provide. Download it [here](https://doi.org/10.17632/mf8cdvzjr7.1), 
and place the corresponding dataset files in the /data folder of each project.

## Installation
Use the following command to install the required dependencies:
```bash
# Install using Conda environment
pip install -r requirements.txt
```

## Training
Each project contains a main.py file. Run this file directly in PyCharm or on a server to perform training, validation, 
and testing. All hyperparameters are set in the file and are consistent with those in the paper. Results may vary 
on different machines, and different hyperparameter settings may yield better results.
```bash
python main.py
```
