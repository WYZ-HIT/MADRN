# MAD residual network (MADRN)

By Yizhen Wang.

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Overview of networkh](#overview)
0. [Getting started](#getting-started)


### Introduction

This is a PyTorch implementation of [A Deep Neural Network Based Method for Magnetic Anomaly Detection](https://doi.org/10.1049/smt2.12084). with datasets and pretrained models. 

### Citation

If you use these models in your research, please cite:

@article{wang2022deep,
  title={A deep neural network based method for magnetic anomaly detection},
  author={Wang, Yizhen and Han, Qi and Zhao, Guanyi and Li, Minghui and Zhan, Dechen and Li, Qiong},
  journal={IET Science, Measurement \& Technology},
  volume={16},
  number={1},
  pages={50--58},
  year={2022},
  publisher={Wiley Online Library}
}

### Overview of network
<div align="center">
  <img src="https://github.com/WYZ-HIT/MADRN/tree/main/figures/architecture.pdf" width="900px">
</div>
<p align="center">
  Figure 1: The architecture of MADRN.
</p>


## Getting started
#### Datasets

Download or generate the datasets like the given datasets form.

The datasets is avaliable at: https://drive.google.com/drive/folders/1p5xO6Ptx5RJnOvCdoH5DmcP3_g1Hu4pA?usp=drive_link

#### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Training and Evaluation

To run MADRN with different parameters in the paper, run this command:

```train
python ./tests/Training_all.py
```
To run the MADRN in the paper, run this command:

```train
python ./tests/Test.py
```
