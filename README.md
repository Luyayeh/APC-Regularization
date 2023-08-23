# APC-Regularization
This project is about experienting the Autoregressive Predictive Coding modeled by LSTM with addtional regularization implmentations such as additive white Gaussian noise (AWGN) and dropout. This project has three parts. 
APC: Speech representation learning model serves as pretraining
FRMCLS: Phone classifeir serves as probing task  
SPKID: Speaker classifeir serves as probing task  
# Dependencies
Python 3.8.10
PyTorch 1.13
Numpy 1.24.1
# Dataset
LibriSpeech: train-clean-100, train-clean-360
WSJ: si284-0.9-train, si284-0.9-dev
Voxceleb1: spkid-train, spkid-dev
# Preprocessing
Generate log Mel spectrograms by Kaldi. The dataset were all preprocessed and granded by Dr. Han Tang, so please reach him out for preprosedd dataset. 
# APC 
Train APC-baseline: run apc/train-epoch.sh  
Train APC-baseline in decay: run apc/train-decay.sh 
Run apc/train-epoch-AWGN.sh to train APC-AWGN
Run apc/train-decay-AWGN.sh to train APC-AWGN in decay
Run apc/train-epoch-dropout.sh to train APC-dropout
Run apc/train-decay-dropout.sh to train APC-dropout in decay
Run apc/train-epoch-AWGN.sh to train APC-AWGN
Run apc/train-decay-AWGN.sh to train APC-AWGN in decay
Run apc/test.sh to test APC 

Train APC-AWGN: run apc/train-epoch-AWGN.sh
Train APC-AWGN in decay: run apc/train-decay-AWGN.sh
Train APC-dropout: run apc/train-epoch-dropout.sh
Train APC-dropout in decay: run apc/train-decay-dropout.sh
Test APC: run apc/test.sh
Train APC-baseline: run apc/train-epoch.sh
Train APC-baseline in decay: run apc/train-decay.sh
