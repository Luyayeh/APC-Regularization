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
Generate log Mel spectrograms by Kaldi. 

The dataset were preprocessed and granded by Dr. Han Tang. 

#  Methodology
Employed an APC model with a 3-layer LSTM architecture, trained on the Librispeech 360-hour dataset. The key regularization techniques used were:

AWGN: Gaussian noise was added to the input features during training to simulate real-world noise and improve the model's robustness

Dropout: Applied between LSTM layers to prevent overfitting, with dropout rates set at 0.2 and 0.5.

# Experiments

The APC model was trained for 20 epochs, followed by 5 epochs with a learning rate decay. Model variants are:

Baseline: No regularization

AWGN: Gaussian noise with SNR10 and SNR30

Dropout: Dropout rates of 0.2 and 0.5

Combined: Both AWGN and dropout

Evaluated employed methodology via two downstream tasks:

Phone Classification: Evaluated using Phone Error Rate (PER) on the WSJ dataset

Speaker Identification: Evaluated using Equal Error Rate (EER) on the VoxCeleb1 dataset

# Results

APC Training Loss: 

Regularization increased the training loss but led to better generalization. The SNR30 model closely matched the baseline, confirming that mild noise does not significantly affect training.

Phone Classification:

AWGN: Improved robustness to noisy inputs, with the SNR10 model consistently outperforming the baseline.

Dropout: Enhanced generalization, with the Dropout0.5 model achieving the best performance.

Combined: The SNR10-Dropout0.5 model balanced improvements in performance across layers.

Speaker Identification:

The speaker identification task did not show significant improvements with regularization, indicating that the benefits of regularization are more pronounced in frame-level tasks such as phone classification.

# APC 
Training configurations: apc/exp/apc-layer3/train.conf

Testing configurations: apc/exp/apc-layer3/test.conf

Train APC-baseline: run apc/train-epoch.sh

Train APC-baseline in decay: run apc/train-decay.sh

Train APC-AWGN: run apc/train-epoch-AWGN.sh

Train APC-AWGN in decay: run apc/train-decay-AWGN.sh

Train APC-dropout: run apc/train-epoch-dropout.sh
Train APC-dropout in decay: run apc/train-decay-dropout.sh

Train APC-AWGN-dropout: run apc/train-epoch-awgn+dropout.sh

Train APC-AWGN-dropout in decay: run apc/train-decay-awgn+dropout.sh

Test APC: run apc/test.sh

Evaluate APC loss: run apc/util/avg-loss.py

# FRMCLS 
Training configurations: frmcls/exp/frmcls-layer3/train.conf

Testing configurations: frmcls/exp/frmcls-layer3/test.conf

Train FRMCLS: run frmcls/train.sh

Validate FRMCLS: run frmcls/test.sh

Validate FRMCLS with random noise: run frmcls/test-noise.sh

Evaluate PER: run frmcls/eval.sh

Confusion matrix: run frmcls/util/confusion-matrix.py

Differentiated confusion matrix : run frmcls/util/confusion-matrix-difference.py

# SPKID
Training configurations: spkid/exp/spkid-layer3/train.conf

Testing configurations: spkid/exp/spkid-layer3/test.conf

Train SPKID: run spkid/train-probe.sh

Validate SPKID and evaluate EER: spkid/test-probe.sh



