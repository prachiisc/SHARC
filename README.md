# Supervised Hierarchical Clustering using Graph Neural Networks for Speaker Diarization
This is the implementation of the following paper:
- Singh, Prachi, et. al. (2023)."Supervised Hierarchical Clustering using Graph Neural Networks for Speaker Diarization." Proceedings of ICASSP 2023.
([paper](https://arxiv.org/pdf/2302.12716.pdf))

## 
- 24-04-2024 : Updated ReadMe and added missing dir
## Overview

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the recipes](#running-the-recipes)
<!-- - [Expected results](#expected-results) -->
- [Cite](#cite)
- [Contact](#contact) 

## Prerequisites

The following packages are required to run the code.

- [Python](https://www.python.org/) >= 3.7
- [Kaldi](https://github.com/kaldi-asr/kaldi)
- [dscore](https://github.com/nryant/dscore)
- [Voxconverse](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)
- [AMI](https://huggingface.co/datasets/edinburghcstr/ami)

## Pretrained Models
The following pretrained models are provided.
- ETDNN x-vector model.
- PLDA models for Voxconverse and AMI dataset.
- SHARC models for Voxconverse and AMI.

## Installation

### Clone the repo and create a new virtual environment

- clone the repo:
```bash
$ git clone git@github.com:prachiisc/SHARC.git
$ cd SHARC
```

- Create the environment: We recommend running the recipes from a fresh virtual environment. 
Make sure to activate the environment before proceeding.

```bash
$ conda create --name SHARC --file requirements.txt
$ conda activate SHARC
```
- Install [Kaldi](https://github.com/kaldi-asr/kaldi). 
If you are a Kaldi novice, please consult the following for additional documentation:
    - [Kaldi tutorial](http://kaldi-asr.org/doc/tutorial.html)
    - [Kaldi for Dummies tutorial](http://kaldi-asr.org/doc/kaldi_for_dummies.html)
<!-- - Go to cloned repository and copy kaldi path in ``path.sh`` given as: -->
- Go to cloned repository and create Softlinks of necessary directories given as:

 <!-- Add "export KALDI_ROOT=/path_of_kaldi_directory/kaldi" in the first line of $local_dir/path.sh -->

 ```sh
$ local_dir="Full_path_of_cloned_repository"
$ KALDI_PATH=/path_of_kaldi_directory/kaldi
$ cd $local_dir
$ ln -sf $KALDI_PATH kaldi
$ . ./path.sh
$ ln -sf kaldi/egs/wsj/s5/steps .  # steps dir
```
- Check the data directories in tools_diar/data
Change tools_diar/data/datasetname/wav.scp with your path of wavfiles.

## Running the recipes

We include full recipes for reproducing the results for Voxconverse and AMI dataset:

### Testing on the Voxconverse dataset

#### Step 1: X-vector extraction, groundtruth label creation and lists directory formation

```bash
   bash services/test_xvec_preprocess.sh <vox_set> nj
```
<vox_set> : vox_diar/vox_diar_test

nj : number of jobs [min(40,number of processors available)]

#### Step 2: Testing

```bash
   bash scripts/test_xvec_parallel.sh Vox
```

### Testing on the AMI dataset
#### Step 1: X-vector extraction, groundtruth label creation and lists directory formation

```bash
   bash services/test_xvec_preprocess.sh <ami_set> nj
```
<ami_set> : ami_dev/ami_eval 

nj : number of jobs [min(15,number of processors available)]

#### Step 2: Testing the SHARC model

```bash
   bash scripts/test_xvec_parallel.sh AMI
```
### Training the model
#### Step 1: X-vector extraction, groundtruth label creation and lists directory formation

```bash
   bash services/test_xvec_preprocess.sh <train_set> <nj>
```

#### Step 2: Training for Voxconverse/AMI

```bash
   bash scripts/train_xvec.sh <Vox/AMI>
```

## Cite
If you are using the resource, please cite as follows: <br />
```
@INPROCEEDINGS{10095372,  
  author={Singh, Prachi and Kaul, Amrit and Ganapathy, Sriram},
  booktitle={2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  title={Supervised Hierarchical Clustering Using Graph Neural Networks for Speaker Diarization},  
  year={2023}, 
  volume={}, 
  pages={1-5}, 
  doi={10.1109/ICASSP49357.2023.10095372}}
  
 ```
## Contact
If you have any comment or question, please contact prachisingh@iisc.ac.in
