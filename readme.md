# Deep Temporal Model for Surgical Phase Recognition

Demo notebook for laparoscopic cholecystectomy phase recognition using a CNN-biLSTM-CRF.

## Description

Laparoscopic cholecystectomy is a surgical procedure for removing a patient's gallbladder. As a minimally invasive procedure it is video-monitored via endoscopic cameras.

Our algorithm analyzes the video recordings from those cameras to automatically identify the **7 surgical phases** making up the procedure:

- Preparation
- Calot triangle dissection
- Clipping and cutting
- Gallbladder dissection
- Gallbladder retraction
- Cleaning and coagulation
- Gallbladder packaging

The underlying deep neural network is a stack of:

- Resnet-50
- Bidirectional LSTM
- Linear-chain CRF

Training was performed on 80 videos from *cholec120*, a superset of the publicly released *cholec80* dataset.

On a test set of 30 videos from *cholec120*, accuracy reaches **89.5%**. Average F1 score over all 7 phases reaches **82.5%**.

## Requirements

- Python 3
- Tensorflow 1.14
- numpy
- opencv 3.4
- matplotlib
- ruamel_yaml

## Citation

*Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition* -  Tong Yu, Didier Mutter, Jacques Marescaux, Nicolas Padoy - IPCAI 2019

[Link](https://arxiv.org/abs/1812.00033)
