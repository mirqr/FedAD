# FedAD

# Anomaly Detection through Unsupervised Federated Learning

This repository contains a Python implementation of the paper "Anomaly Detection through Unsupervised Federated Learning" by Mirko Nardi, Lorenzo Valerio, and Andrea Passarella.

## Overview

Federated learning (FL) enables a set of clients to collaboratively train a machine learning model while keeping the data decentralized. This paper extends the FL paradigm to unsupervised tasks by addressing the problem of anomaly detection in decentralized settings.

A method is in which clients are grouped into communities, each having similar majority (i.e., inlier) patterns. Each community of clients trains the same anomaly detection model (i.e., autoencoders) in a federated fashion. The resulting model is then shared and used to detect anomalies within the clients of the same community that joined the corresponding federated process.

## Installation

This code is tested on Python 3.9. To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage

To run the code, set a configuration in the runconfig.py file and run the Jupyter notebooks in the following order:

1. **step_1.ipynb**: This notebook contains the code for the community detection phase.
2. **step_2.ipynb**: This notebook contains the code for the federated anomaly detection phase.




## Citation

If you use this code in your research, please cite the original paper:

Nardi, M., Valerio, L., & Passarella, A. (2022). Anomaly Detection through Unsupervised Federated Learning. arXiv preprint arXiv:2209.04184.

You can find a link to the paper [here](https://arxiv.org/abs/2209.04184).

## License

This project is licensed under the MIT License.
