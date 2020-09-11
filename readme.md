<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="visuals/camma_logo_tr.png" width="20%">
</a>
</div>

# Deep Temporal Model for Surgical Phase Recognition

Demo notebook for laparoscopic cholecystectomy phase recognition using a CNN-biLSTM-CRF.
-------------------
**Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition (IPCAI 2019)**

_Tong Yu, Didier Mutter, Jacques Marescaux, Nicolas Padoy_

[![arXiv](https://img.shields.io/badge/arxiv-1812.00033-red)](https://arxiv.org/abs/1812.00033)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CAMMA-public/Surgical-Phase-Recognition/blob/master/phase_recognition_demo_colab.ipynb)

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

![model](visuals/model.png)

Training was performed on 80 videos from *cholec120*, a superset of the publicly released *cholec80* dataset available [here](http://camma.u-strasbg.fr/datasets).

On a test set of 30 videos from *cholec120*, accuracy reaches **89.5%**. Average F1 score over all 7 phases reaches **82.5%**.

## Requirements

- Python 3
- Tensorflow 1.14
- numpy
- opencv 3.4
- matplotlib
- ruamel_yaml

Developer configuration info:
- Ubuntu 20.04
- CUDA 10.1
- NVIDIA GTX1080Ti GPU

## TF-Cholec80

A version of the notebook using [TF-Cholec80]() is also available (`phase_recognition_demo_tfc.ipynb`).

## Citation
```bibtex
@inproceedings{yu2019surgicalphase,
title = {Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition},
author = {Tong Yu, Didier Mutter, Jacques Marescaux, Nicolas Padoy},
booktitle = {International Conference on Information Processing in Computer-Assisted Interventions},
year = {2019}
}
```

## License
This code may be used for **non-commercial scientific research purposes** as defined by [Creative Commons 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.
