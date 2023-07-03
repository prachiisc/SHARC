This is the implementation of the following paper:
- Singh, Prachi, et. al. (2023)."SUPERVISED HIERARCHICAL CLUSTERING USING GRAPH NEURAL NETWORKS FOR SPEAKER DIARIZATION." Proceedings of ICASSP 2023.
([paper]())

# Overview

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the recipes](#running-the-recipes)
- [Expected results](#expected-results)
- [Reproducibility](#reproducibility)

# Prerequisites

The following packages are required to run the code.

- [Python](https://www.python.org/) >= 3.7
- [Kaldi](https://github.com/kaldi-asr/kaldi)
- [dscore](https://github.com/nryant/dscore)
- [Voxconverse]
- [AMI] 

# Inclusion
The following models are provided.
- ETDNN x-vector model.
- PLDA models for Voxconverse and AMI dataset.
- SHARC pre-trained models for Voxconverse and AMI.

# Installation

## Clone the repo and create a new virtual environment

Clone the repo:

```bash
git clone 
cd SHARC
```
We recommend running the recipes from a fresh virtual environment. 
Make sure to activate the environment before proceeding.

```bash
conda create --name SHARC --file requirements.txt
conda activate SHARC
```

# Running the recipes

We include full recipes for reproducing the results for Voxconverse and AMI dataset:

## Running the Voxconverse recipe

## Step 1: X-vector extraction, groundtruth label creation and lists directory formation

```bash
   bash services/
```

## Step 2: Training the model (skip if this step)



## Running the AMI recipe